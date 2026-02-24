import os
import argparse
import subprocess

def run_command(cmd):
    print("\nRunning:", cmd)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError("command failed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--model", required=True, choices=["rcnn", "yolo"])
    parser.add_argument("--method", default="IR", choices=["IR", "LR"])
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--class_agnostic", action="store_true")
    args = parser.parse_args()
    os.makedirs(f"{args.model}/final_detections", exist_ok=True)
    os.makedirs("/models/calibrations", exist_ok=True)
    detection_json = f"{args.model}/final_detections/{args.model}_val_bbox.json"
    calibrator_output = f"/models/calibrations/{args.model}.pkl"

    detect_cmd = f"""python generate_detections_bdd.py --image_dir {args.image_dir} --output {detection_json} --model {args.model}"""
    run_command(detect_cmd)
    class_flag = "--class_agnostic" if args.class_agnostic else ""
    calibrate_cmd = f"""python calibrate_bdd.py --image_dir {args.image_dir} --label_dir {args.label_dir} --detections {detection_json} --output {calibrator_output} --type {args.method} {class_flag} --num_images {args.num_images}"""
    run_command(calibrate_cmd)
    print("calibrator saved in this path", calibrator_output)