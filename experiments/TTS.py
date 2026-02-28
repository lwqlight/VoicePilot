import os
import time
import argparse
import logging

try:
    import torch
except Exception:
    torch = None


logger = logging.getLogger("tts")


def set_thread_env(threads: int):
    """Set environment variables commonly used by BLAS/OpenMP and torch thread counts."""
    if threads is None or threads <= 0:
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))
    if torch is not None:
        try:
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(max(1, threads // 2))
        except Exception:
            pass


def load_model(model_dir: str, device: str):
    # Delay import so the script can show --help without funasr installed
    from funasr import AutoModel

    remote_code = os.path.join(model_dir, "model.py")
    if not os.path.isfile(remote_code):
        remote_code = None
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code=remote_code,
        device=device,
    )
    return model


def infer(model, input_path: str, language: str, use_itn: bool, batch_size_s: int):
    return model.generate(
        input=input_path,
        language=language,
        use_itn=use_itn,
        batch_size_s=batch_size_s,
    )


def main():
    parser = argparse.ArgumentParser(description="Lightweight TTS runner with simple optimizations for Raspberry Pi")
    parser.add_argument("--model", default="./SenseVoiceSmall", help="model directory")
    parser.add_argument("--input", default="./example/edgetts1.mp3", help="input audio or text file")
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="device to run on (cpu only for Raspberry Pi)")
    parser.add_argument("--threads", type=int, default=4, help="number of CPU threads to use")
    parser.add_argument("--batch_size_s", type=int, default=60, help="batch size seconds for generation")
    parser.add_argument("--warmup", type=int, default=1, help="number of warmup runs to reduce variance")
    parser.add_argument("--iters", type=int, default=1, help="number of timed iterations to run when benchmarking")
    parser.add_argument("--bench", action="store_true", help="print benchmark timings")
    parser.add_argument("--use_itn", action="store_true", default=True, help="whether to use ITN (inverse text normalization)")
    parser.add_argument("--no-use_itn", dest="use_itn", action="store_false", help="disable ITN")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info("Setting thread environment to %s", args.threads)
    set_thread_env(args.threads)

    logger.info("Loading model from %s on %s", args.model, args.device)
    model = load_model(args.model, device=args.device)

    # Warm-up runs to stabilize performance numbers
    if args.warmup > 0:
        logger.info("Warming up (%d runs)...", args.warmup)
        for i in range(args.warmup):
            try:
                _ = infer(model, args.input, language="auto", use_itn=args.use_itn, batch_size_s=args.batch_size_s)
            except Exception as e:
                logger.warning("Warm-up run %d failed: %s", i + 1, e)

    # Timed iterations
    times = []
    results = None
    for i in range(max(1, args.iters)):
        t0 = time.perf_counter()
        try:
            results = infer(model, args.input, language="auto", use_itn=args.use_itn, batch_size_s=args.batch_size_s)
        except Exception as e:
            logger.error("Inference failed: %s", e)
            raise
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        logger.info("Iteration %d completed in %.3f s", i + 1, elapsed)

    if args.bench:
        import statistics

        mean = statistics.mean(times)
        median = statistics.median(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        logger.info("Benchmark results: mean=%.3f s, median=%.3f s, stdev=%.3f s", mean, median, stdev)

    if results:
        try:
            print(results[0]["text"])
        except Exception:
            logger.info("Result format unexpected, raw output:\n%s", results)


if __name__ == "__main__":
    main()
