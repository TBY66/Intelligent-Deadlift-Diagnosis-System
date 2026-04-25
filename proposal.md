# Abstract
I'm creating an intelligent deadlift diagnosis system. There would be four versions. In short, the first version is an GUI application that works on MacOS which enables user to upload a video, and then the system automatically detects reptitions, classfies movement pattern of every rep, and eventually gives a feedback based on the results. The second version is almost the same, the application is packaged into .exe so that it works on Windows OS. The only difference between the third version and the first/second version is that the third version works on iPhone Safari. It records the movement with camera and analyzes it on the website. The third version is the final version. It makes the website compatible on Safari, Chrome or any other Android browsers.

# Version 1.0
- Merge the code gracefully, so that every YOLO model runs only one time on a video when parsing. Don't make the code reduntant.
- Materials are in `materials/`.
- The layouts are in `layout_status1.png`, `layout_status2.png`,`layout_status3.png`.
## Upload block
- Enables user to drag file from outside the window or click "upload video" to upload video directly.
## Video block
- Shows the uploaded video.
- Enables user to play, pause, stop or drag the timebar.
## Parse video
1. Use `yolo_models/yolov11s-pose.pt` to detect the largest person by bounding box area. We only extract the keypoints' x, y coordinates and back features (detected by `yolo_models/yolov11s-seg.pt`) from this person.
2. Use `yolo_models/yolov11s-gymequipment.pt` to detect gym equipments in the video. Detect only 5: Barbell, 9: Dumbbell, 10: Gymball, 12: Kettlebell, which is defined in `raw_py/data.yaml`. It may detect multiple objects, but only the equipment that is closest to the person's both wrists and is moving in an upward and downward pattern is our interest.
3. A video may include multiple repetitions. In `raw_py/crop_video.py`, the algorithm is to detect the y-coordinate of "Neck" marker in `.trc` file and crop the video into continuous clips by analyzing y-coordinate minima (valleys). The first clip starts from the y-coordinate valley of the selected time stamp. Now, I would like to make the algorithm work on YOLO-Pose and even more robust. Use `yolo_models/yolov11s-pose.pt` to detect "nose" as a alternative of "Neck". In the meanwhile, use `yolo_models/yolov11s-gymequipment.pt` to detect the y-coordinate of the center of the object. The start timestamp of the first clip is defined as "the first y-coordinate valley AND within a time range the equipment starts moving vertically"
## Extract features
4. Use `raw_py/extract_keypoint.py` to record the keypoints' x, y coordinates from the sampled 20 frames from the clip.
5. Use `raw_py/extract_backfeatures.py` to record the back features: k, o, v from the sampled 20 frames from the clip. 
6. Use `raw_py/build_inputdata.py` to build data to input the classification models.
## Inference
7. Use the inference phase of `raw_py/train_model.py` to classify movement of each clip. Use "fold5" tf1 ,tf2 and xgboost model in `models52/`
8. Each clip has a 4-dimension vector [c, h, k, r] with binary values showing the predicted results. Record results of every clip.
## Analysis block
- Output the fraction of occurence of each class, e.g. the user performed 10 reps, the results shows:  
Correct: 2/10   
Hip first: 3/10  
Rounded back: 3/10  
Don't output the class that never occured
## Rule-based feedback block
- Feedback rules:
- Correct: You did it well!
- Hip first: Your hips rose to early. Focus on driving through your legs and raise your chest simultaneously.
- Knee dominant: You push your knees too forward and your trunk too upright. Work on hinging your hips.
- Rounded back: Brace the core and tighten your lower back.
1. If total reps = 1:  
Print the corresponding feedbacks. (May include multiple errors)
2. If total reps = 2:  
if both reps correct: Print "You did both reps well!"
elseif one rep is correct: Print "You did the {int: rep_id} well, but the other rep + {feedback}."
else(both reps error): Print "For the first rep, {feedback}. For the second rep, {feedback}.
3. If total reps >= 3: break the sequence into three parts, "beginning", "middle", "end".
Do major voting for every part, so that every part has a "part result" [c, h, k, r].
if all parts correct: Print "You did well on all reps!"
elseif some parts but not all parts correct: Print "You did well in the {part} (and in the {part}), but for the other repetitions, {feedback}.
else(all parts are not correct):  
if all three parts do not have common error: Print "In the beginning, {feedback}. In the middle, {feed back}. In the end, {feedback}."
elseif some parts have common error: Print "In the {part} (and in the {part}), {feedback}."
else(all three parts have the same error): Print "For the whole set, {feedback}."
# Upload history block
- Temporarily stores the video and the analysis results.
- Shows the original file name as default
- On the right of the file name there is a "three-stripe" icon, that enables user to "anaylze", "rename" or "delete" file.
- Enables user to rename file name by double clicking.
- Enables user to drag file up or down to reoder in the list.
- Whenever user single click the file, the analyzed results are shown on the left.
- A "Analyze all" button, and a "Delete all" button at the bottom of the block.