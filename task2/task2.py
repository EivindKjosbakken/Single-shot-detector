from distutils.file_util import copy_file
import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    xmin1, xmin2 = prediction_box[0], gt_box[0]
    ymin1, ymin2 = prediction_box[1], gt_box[1]
    xmax1, xmax2 = prediction_box[2], gt_box[2]
    ymax1, ymax2 = prediction_box[3], gt_box[3]
 
    area1 = (xmax1-xmin1)*(ymax1-ymin1)
    area2 = (xmax2-xmin2)*(ymax2-ymin2)
    # Compute intersection
    a = max(xmin1, xmin2) #get dimensions of the intersection "fbox"
    b = max(ymin1, ymin2)
    c = min(xmax1, xmax2)
    d = min(ymax1, ymax2)
    if (a>=c or b>=d): #if there is no intersection, return 0
        return 0
    intersection = (c-a)*(d-b)
    # Compute union
    union = area1+area2-intersection #formula for union
    
    iou = intersection/union #formula for iou

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) == 0:
        return 1
    return float((num_tp/(num_tp+num_fp)))


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn) == 0:
        return 0
    return float((num_tp/(num_tp+num_fn)))


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    #if one of the arrays are empty, then there is no match ofcourse
    if (prediction_boxes.size == 0 or gt_boxes.size==0):
        return prediction_boxes, gt_boxes

    # Find all possible matches with a IoU >= iou threshold
    predBoxes = []
    for predBox in prediction_boxes:
        for gtBox in gt_boxes:
            iou = calculate_iou(predBox, gtBox)
            if (iou>= iou_threshold):
                predBoxes.append([predBox,iou]) #storing iou so I don't have to calculate again
   
    predBoxes = sorted(predBoxes, key=lambda x: x[1], reverse=True) #sort on iou, decreasing
    npPredBoxes = np.array(predBoxes, dtype=object)
    
    # Find all matches with the highest IoU threshold
    predBoxResult = []
    gtBoxResult = []
    for npPredBox in npPredBoxes:
        foundMatch = False
        for gtBox in gt_boxes: 
            if (calculate_iou(np.array(npPredBox[0]), np.array(gtBox)) >= iou_threshold):
                predBoxResult.append(npPredBox[0])
                gtBoxResult.append(gtBox)
                foundMatch = True
                break #go to next gt box, as this gt box got a match
        if not foundMatch:
            predBoxResult.append(None) #if no match over the threshold
    return np.array(predBoxResult), np.array(gtBoxResult)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, "false_neg": int}
    """
    res = dict() #result is getting stored in dictionary
    predBoxes, gtBoxes = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold) #matching the boxes
    res["true_pos"] = len(predBoxes)
    res["false_pos"] = len(prediction_boxes) - res["true_pos"]
    res["false_neg"] = len(gt_boxes) - res["true_pos"]
    return res

def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    truePos = 0
    falsePos = 0
    falseNeg = 0
    for i in range(len(all_prediction_boxes)):
        ans = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        truePos+=ans["true_pos"]
        falsePos+=ans["false_pos"]
        falseNeg+=ans["false_neg"]

    precision = calculate_precision(truePos, falsePos, falseNeg)
    recall = calculate_recall(truePos, falsePos, falseNeg)
    return (precision, recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
 
    precisions = [] 
    recalls = []
    
    for confidence_threshold in confidence_thresholds:
        predBoxesPerThreshold = [] #a list for each threshhold
        for i in range(len(confidence_scores)):
            currPredBoxes = []
            for j in range(len(confidence_scores[i])):
                if (confidence_scores[i][j] >= confidence_threshold): #if over score, we use the image to calc precision/recall
                    currPredBoxes.append(all_prediction_boxes[i][j])
            
            predBoxesPerThreshold.append(np.array(currPredBoxes))
        
        precision, recall = calculate_precision_recall_all_images(predBoxesPerThreshold, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)
    

def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    totalPrecision = 0
    totPrec = []
    for level in recall_levels:
        precMax = 0
        for i in range(len(precisions)):
            if (round(recalls[i], 3) >= round(level, 5) and (precisions[i] > precMax)): #rounding because of python floating point error
                #print("level: ", round(level, 5), "prec: ", precisions[i])
                precMax=precisions[i]
        totalPrecision+=(precMax)
        totPrec.append(precMax)
    #print(totPrec)
    average_precision = np.mean(totPrec)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """

    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
