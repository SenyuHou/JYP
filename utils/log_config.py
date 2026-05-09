import os
import logging
import time
import json


def setup_logger(args):
    log_path = os.path.join(os.getcwd(), args.log_path)
    log_path = os.path.join(log_path, f"{args.dataset}")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if args.dataset in ["cifar10", "cifar100"]:
        log_file_name = f"{args.noise_type}-{args.noise_ratio}-{timestamp}-{args.alg_name}.log"
    else:
        log_file_name = f"{timestamp}-{args.alg_name}.log"

    log_file_path = os.path.join(log_path, log_file_name)

    logger = logging.getLogger("training_log")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    args_str = json.dumps(vars(args), indent=4)
    logger.info("Parameters:\n" + args_str)

    return logger
