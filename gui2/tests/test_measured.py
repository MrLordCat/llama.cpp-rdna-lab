"""Reading a run's memory back out of its own log.

Every line quoted here is verbatim from a real launch: a Qwen3.8-27B split
across two local cards and one RPC worker, with a speculative draft context.
That run is the whole reference, so if llama.cpp ever changes how it announces
a buffer, these are what should notice.
"""

from __future__ import annotations

from gui2.core.measured import DeviceUse, is_accelerator, notes, parse_text, split_name

LOAD = """\
load_tensors:   CPU_Mapped model buffer size =   682.03 MiB
load_tensors: RPC0[192.168.1.60:50052] model buffer size =  6011.49 MiB
load_tensors:      Vulkan0 model buffer size =  4615.84 MiB
load_tensors:      Vulkan1 model buffer size =  5035.55 MiB
llama_context:        CPU  output buffer size =     0.95 MiB
llama_kv_cache: RPC0[192.168.1.60:50052] KV buffer size =  2304.00 MiB
llama_kv_cache:    Vulkan0 KV buffer size =  1380.00 MiB
llama_kv_cache:    Vulkan1 KV buffer size =  1020.00 MiB
llama_memory_recurrent: RPC0[192.168.1.60:50052] RS buffer size =   233.79 MiB
llama_memory_recurrent:    Vulkan0 RS buffer size =   249.38 MiB
llama_memory_recurrent:    Vulkan1 RS buffer size =   264.96 MiB
"""

#: the draft context llama-server builds after the main one. Its buffers are
#: live memory for the whole run, and the shutdown table leaves them out.
DRAFT = """\
llama_context:        CPU  output buffer size =     0.95 MiB
llama_kv_cache: RPC0[192.168.1.60:50052] KV buffer size =   384.00 MiB
srv    load_model: speculative draft context initialized, n_ctx = 98304
"""

BREAKDOWN = """\
common_memory_breakdown_print: | memory breakdown [MiB]        | total             free    self   model   context   compute    unaccounted |
common_memory_breakdown_print: |   - Vulkan1 (RX 9070 XT)      | 16304 =           8721 + (6341 =  5035 +    1284 +      20) +        1241 |
common_memory_breakdown_print: |   - Vulkan0 (RX 9070 XT)      | 16304 =           8817 + (6264 =  4615 +    1629 +      19) +        1221 |
common_memory_breakdown_print: |   - RPC0 (192.168.1.60:50052) | 10267 = 17592186044402 + (8569 =  6011 +    2537 +      19) +        1711 |
common_memory_breakdown_print: |   - Host                      |                            688 =   682 +       0 +       6                |
"""

FULL = LOAD + DRAFT + BREAKDOWN


def test_the_model_mapped_into_ram_is_not_video_memory():
    # the one mistake that costs gigabytes: CPU_Mapped is the file on disk,
    # mapped, and Vulkan_Host is pinned system memory. Neither is on a card.
    assert is_accelerator("Vulkan0")
    assert is_accelerator("ROCm1")
    assert is_accelerator("RPC0[192.168.1.60:50052]")
    assert not is_accelerator("CPU_Mapped")
    assert not is_accelerator("CPU")
    assert not is_accelerator("Vulkan_Host")
    assert not is_accelerator("Vulkan_Host_Direct")
    assert not is_accelerator("Host")


def test_the_host_buffers_stay_out_of_the_vram_total():
    measurement = parse_text(FULL)

    assert {device.name for device in measurement.vram} == {"Vulkan0", "Vulkan1", "RPC0"}
    assert {device.name for device in measurement.host} == {"CPU", "CPU_Mapped"}
    # 682.03 of mapped weights and 1.90 of output would have been counted twice
    # over as video memory by a parser that only looked for "buffer size"
    assert round(measurement.vram_mib, 2) == round(6264.22 + 6340.51 + 8952.28, 2)


def test_an_rpc_worker_is_one_device_under_two_spellings():
    # the allocation lines carry the endpoint inside the name, the shutdown
    # table carries it beside the name, and --rpc only ever says RPC0
    assert split_name("RPC0[192.168.1.60:50052]") == ("RPC0", "192.168.1.60:50052")
    assert split_name("Vulkan0") == ("Vulkan0", "")

    worker = parse_text(FULL).find("RPC0")
    assert worker is not None
    assert worker.description == "192.168.1.60:50052"


def test_the_draft_context_is_charged_to_the_card_that_holds_it():
    measurement = parse_text(FULL)
    worker = measurement.find("RPC0")

    # 2304 for the main cache plus 384 for the draft: both are resident
    assert worker.kv_mib == 2688.00
    # which is why the figure sits above what the shutdown table owns up to
    assert worker.used_mib > 8569


def test_a_local_card_reproduces_its_own_shutdown_table():
    card = parse_text(FULL).find("Vulkan0")

    assert card.model_mib == 4615.84
    assert card.kv_mib == 1380.00
    assert card.state_mib == 249.38
    assert round(card.used_mib) == 6264  # the table's own self column
    assert card.total_mib == 16304
    assert card.overhead_mib == 1221.0


def test_a_compute_buffer_nobody_announced_comes_from_the_table():
    # this build prints no compute buffer line at load; the only place the
    # figure appears is the shutdown table, so it has to be taken from there
    assert "compute buffer size" not in LOAD
    assert parse_text(FULL).find("Vulkan1").compute_mib == 20.0


def test_the_destructors_complaint_is_not_an_allocation():
    noise = (
        "~llama_context:      ROCm0 compute buffer size of   3.5024 MiB, "
        "does not match expectation of 130.4102 MiB\n"
    )
    measurement = parse_text(LOAD + noise)

    assert measurement.find("ROCm0") is None


def test_a_log_still_being_written_says_so():
    measurement = parse_text(LOAD)

    assert measurement
    assert not measurement.complete
    assert measurement.find("Vulkan0").total_mib is None
    assert any("compute buffers" in note for note in notes(measurement))

    assert parse_text(FULL).complete
    assert not any("compute buffers" in note for note in notes(parse_text(FULL)))


def test_a_restart_in_one_log_reports_the_run_that_finished_last():
    later = BREAKDOWN.replace("(6264 =  4615 +    1629 +      19) +        1221", "(6264 =  4615 +    1629 +      19) +          17")
    measurement = parse_text(FULL + later)

    # the second table replaces the first rather than adding to it
    assert measurement.find("Vulkan0").overhead_mib == 17.0


def test_the_pieces_come_back_largest_first():
    card = parse_text(FULL).find("Vulkan1")

    assert card.parts == (
        ("weights", 5035.55),
        ("KV cache", 1020.00),
        ("recurrent state", 264.96),
        ("compute", 20.0),
    )


def test_a_log_with_nothing_in_it_measures_nothing():
    measurement = parse_text("main: server is listening on http://127.0.0.1:8080\n")

    assert not measurement
    assert measurement.vram_mib == 0
    assert list(notes(measurement)) == []


def test_a_quiet_card_is_reported_without_a_note():
    small = DeviceUse(name="Vulkan0", model_mib=1024.0, overhead_mib=64.0)

    assert list(notes(parse_text(LOAD + BREAKDOWN))) != []
    assert small.is_vram
    assert small.used_mib == 1024.0
