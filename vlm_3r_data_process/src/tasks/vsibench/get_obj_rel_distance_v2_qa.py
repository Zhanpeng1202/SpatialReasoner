"""
The format of the qa file is as follows:
{
    "id": id,
    "scene_name": "xxxx",
    "question_type": "the question type a.k.a. the task at hand",
    "video_path": "path to the video",
    "category: "category of the object",
    "question": "question",
    "options": "answer options",
    "ground truth": "either the float or string answer",
    "mc_answer": "letter answer",
}
"""

import random
import itertools

from src.tasks.base_qa_generator import BaseQAGenerator
from src.utils.common_utils import cal_3d_bbox_distance_between_categories, from_options_to_mc_answer


def ambiguous(distances, threshold=0.15):
    """
    Check if any of the distance answers are within threshold of each other, meaning that the ground truth answer would likely be ambiguous to a human annotator.

    Args:
        distances: list[float] of the distances of each option to the queried object
        threshold: float, the distance threshold to check against (default is 0.15 meters).

    Returns:
        bool: True if any pair of objects are within the threshold distance, False otherwise.
    """
    num_choices = len(distances)
    # Compare each pair of choices
    for i in range(num_choices):
        for j in range(i + 1, num_choices):
            if abs(distances[i] - distances[j]) <= threshold:
                return True
    
    return False


def get_gt(choice_categories, primary_object_category, obj_bbox_info, 
           dist_lookup, room_size, answer_counts, option_letters):
    '''
    Arg(s):
        choice_categories: list[str] of chosen object choice categories
        primary_object_category: category of the primary object (to which we are trying to choose the closest object)
        obj_bbox_info: dictionary of object bounding box information
        dist_lookup: dictionary to cache calculated distances
        room_size: float, size of the room
        answer_counts: dictionary to track answer counts for each letter
        option_letters: list of option letters (e.g., ['A', 'B'])

    Output:
        ground_truth: the letter (A, B, C, D) of the correct answer
    '''

    distances = []
    for category in choice_categories:
        if (category, primary_object_category) in dist_lookup:
            dist = dist_lookup[(category, primary_object_category)]
        elif (primary_object_category, category) in dist_lookup:
            dist = dist_lookup[(primary_object_category, category)]
        else:
            dist = cal_3d_bbox_distance_between_categories(
                obj_bbox_info[category], obj_bbox_info[primary_object_category]
            )
            dist_lookup[(category, primary_object_category)] = dist
            dist_lookup[(primary_object_category, category)] = dist
            
        distances.append(dist)
        
    min_distance = min(distances)

    distance_to_object_threshold = 0.15
    if min_distance < distance_to_object_threshold:
        return "ambiguous", None, None, answer_counts

    min_index = distances.index(min_distance)

    # check if any of the point cloud instance lists were None
    if min_distance < 0:
        return "point cloud error", None, None, answer_counts
    
    threshold = 0.30 if room_size > 40 else 0.15
    if ambiguous(distances, threshold=threshold): # Pass the correct threshold to ambiguous
        return "ambiguous", None, None, answer_counts
    
    ground_truth = choice_categories[min_index]
    
    # Shuffle the choices
    options, mc_answer, answer_counts = from_options_to_mc_answer(
        choice_categories, ground_truth, answer_counts, option_letters
    )

    return ground_truth, mc_answer, options, answer_counts


class ObjRelDistV2QAGenerator(BaseQAGenerator):
    def __init__(self):
        super().__init__()
        self.answer_counts = {'A': 0, 'B': 0, 'C': 0}  # 3 options
        self.option_letters = ['A', 'B', 'C']

    def get_default_args(self):
        return {
            'question_template': "OBJ_REL_DISTANCE_V2_TEMPLATE",
            'num_subsample': 6,
            'question_type': 'object_rel_distance_v2',
            'output_filename_prefix': 'qa_obj_rel_distance_v2'
        }

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """Generate 3-choice relative distance QA pairs for a single scene."""
        scene_qa_candidates = [] # Candidates before filtering/processing
        video_path = scene_info['video_path']
        obj_bbox_info = scene_info['object_bboxes']
        room_size = scene_info['room_size']
        
        single_count_objects = []
        multiple_count_objects = []
        for category, count in scene_info['object_counts'].items():
            if count == 1:
                single_count_objects.append(category)
            else:
                multiple_count_objects.append(category)

        all_objects_set = set(single_count_objects) | set(multiple_count_objects)

        # Need at least 1 single-count obj and 3 others for 3 choices
        if len(all_objects_set) < 4 or not single_count_objects:
            return [] 

        for primary_category in single_count_objects:
            other_objects = list(all_objects_set - {primary_category})
            # Not enough objects to form 3 choices
            if len(other_objects) < 3:
                continue 
                
            # Generate combinations for multiple choice options
            combinations = list(itertools.combinations(other_objects, 3)) # 3 choices
            for combo in combinations:
                scene_qa_candidates.append({
                    "primary_category": primary_category,
                    "choice_categories": list(combo)
                })

        # Subsample *before* calculating distances and generating QA dicts
        if len(scene_qa_candidates) > self.args.num_subsample:
            scene_qa_candidates = random.sample(scene_qa_candidates, self.args.num_subsample)

        scene_qa_list = []
        dist_lookup = {} # Cache distances within the scene
        for candidate in scene_qa_candidates:
            primary_cat = candidate["primary_category"]
            choice_cats = candidate["choice_categories"]
            
            gt, mc_answer, options, self.answer_counts = get_gt(
                choice_cats, primary_cat, obj_bbox_info, dist_lookup,
                room_size, self.answer_counts, self.option_letters
            )

            if gt == "ambiguous" or gt == "point cloud error":
                continue

            qa = {
                # "id" will be assigned by the base class run method
                "scene_name": scene_name,
                'dataset': self.args.dataset,
                "question_type": self.args.question_type,
                "video_path": video_path,
                "category": primary_cat,
                "question": self.question_template.format( 
                    choice_a=options[0],
                    choice_b=options[1],
                    choice_c=options[2],
                    category=primary_cat
                ),
                "options": [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}"], # 3 options
                "ground_truth": gt, # The category name of the closest object
                "mc_answer": mc_answer
            }
            scene_qa_list.append(qa)
        
        return scene_qa_list


if __name__ == '__main__':
    generator = ObjRelDistV2QAGenerator()
    generator.run() 