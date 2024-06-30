import torch
from dataset import test_dataset,select_classes,select_class_rgb_values,test_dataset_vis
import numpy as np
from utilis import colour_code_segmentation,reverse_one_hot,visualize

model = torch.load('assets/best_model.pth')

model = model.to('cpu')


for idx in range(len(test_dataset)):

    image, gt_mask = test_dataset[idx]
    # image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
    image_vis = test_dataset_vis[idx][0].astype('uint8')
    x_tensor = torch.from_numpy(image).to('cpu').unsqueeze(0)
    # Predict test image
    pred_mask = model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    # Convert pred_mask from `CHW` format to `HWC` format
    pred_mask = np.transpose(pred_mask,(1,2,0))
    # Get prediction channel corresponding to road
    pred_lungs_heatmap = pred_mask[:,:,select_classes.index('lungs')]
    # pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))
    pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
    # Convert gt_mask from `CHW` format to `HWC` format
    gt_mask = np.transpose(gt_mask,(1,2,0))
    # gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values))
    gt_mask = colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values)
    # cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])

    visualize(
        # original_image = image_vis,
        original_image = image_vis,
        ground_truth_mask = gt_mask,
        predicted_mask = pred_mask,
        predicted_lungs_heatmap = pred_lungs_heatmap
    )
