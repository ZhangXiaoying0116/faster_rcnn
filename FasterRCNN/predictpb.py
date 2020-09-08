#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import itertools
import numpy as np
import os
import shutil
import tensorflow as tf
import cv2
import tqdm
import time

import tensorpack.utils.viz as tpviz
from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
from tensorpack.utils import fs, logger

from common import CustomResize, clip_boxes
from dataset import DatasetRegistry, register_coco, register_balloon
from config import config as cfg
from config import finalize_configs
from data import get_eval_dataflow, get_train_dataflow
from eval import DetectionResult, multithread_predict_dataflow, predict_image
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from viz import (
    draw_annotation, draw_final_outputs, draw_predictions,
    draw_proposal_recall, draw_final_outputs_blackwhite)


def do_visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=SmartInit(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/scores'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'fastrcnn_all_scores',
            'output/boxes',
            'output/scores',
            'output/labels',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df), nr_visualize):
            img, gt_boxes, gt_labels = dp['image'], dp['gt_boxes'], dp['gt_labels']

            rpn_boxes, rpn_scores, all_scores, \
                final_boxes, final_scores, final_labels = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_scores[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()


def do_evaluate(pred_config, output_file):
    num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_tower))).get_predictors()

    for dataset in cfg.DATA.VAL:
        logger.info("Evaluating {} ...".format(dataset))
        dataflows = [
            get_eval_dataflow(dataset, shard=k, num_shards=num_tower)
            for k in range(num_tower)]
        all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        output = output_file + '-' + dataset
        DatasetRegistry.get(dataset).eval_inference_results(all_results, output)


def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func)
    if cfg.MODE_MASK:
        final = draw_final_outputs_blackwhite(img, results)
    else:
        final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite("output.png", viz)
    logger.info("Inference output for {} written to output.png".format(input_file))
    # tpviz.interactive_imshow(viz)

def predict_with_pb(model_file,input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (512,512))
    print("img.shape!!!!",img.shape)
    ##preprocess
    # orig_shape = img.shape[:2]
    # resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    # resized_img = resizer.augment(img)
    # scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input_image_tensor = sess.graph.get_tensor_by_name("image:0")
            print("input_image_tensor!!!!!!!!",input_image_tensor)
            output_tensor_boxes = sess.graph.get_tensor_by_name("output/boxes:0")
            output_tensor_scores = sess.graph.get_tensor_by_name("output/scores:0")
            output_tensor_labels = sess.graph.get_tensor_by_name("output/labels:0")
            output= sess.run([output_tensor_boxes,output_tensor_scores,output_tensor_labels], feed_dict={input_image_tensor: img})

            time_start =time.time() 
            for i in range(100):                  
                output= sess.run([output_tensor_boxes,output_tensor_scores,output_tensor_labels], feed_dict={input_image_tensor: img})
            time_end =time.time()
            print('Inference fps:',(time_end-time_start)/100)
            # Some slow numpy postprocessing:
            boxes=output[0]
            probs=output[1]
            labels=output[2]

            # boxes = boxes / scale
            # # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
            # boxes = clip_boxes(boxes, orig_shape)
            # if masks:
            #     full_masks = [_paste_mask(box, mask, orig_shape)
            #                 for box, mask in zip(boxes, masks[0])]
            #     masks = full_masks
            # else:
                # fill with none
            masks = [None] * len(boxes)

        results = [DetectionResult(*args) for args in zip(boxes, probs, labels.tolist(), masks)]

        final = draw_final_outputs(img, results)
        viz = np.concatenate((img, final), axis=1)
        cv2.imwrite("output1.png", viz)
        logger.info("Inference output for {} written to output1.png".format(input_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--load', help='load a model for evaluation.', required=True)
    parser.add_argument('--predict_with_pb', help='infer with pb.')
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file", nargs='+')
    parser.add_argument('--benchmark', action='store_true', help="Benchmark the speed of the model + postprocessing")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')                     
    parser.add_argument('--output-pb', help='Save a model to .pb')
    parser.add_argument('--output-serving', help='Save a model to serving file')

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)
    register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    register_balloon(cfg.DATA.BASEDIR)

    
    if not tf.test.is_gpu_available():
        from tensorflow.python.framework import test_util
        assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
            "Inference requires either GPU support or MKL support!"
    # assert args.load
    finalize_configs(is_training=False)

    if args.predict or args.visualize:
        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    if args.predict_with_pb:
        for image_file in args.predict:
            predict_with_pb(args.predict_with_pb,image_file)
    # if args.load:   
    #     MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()
    #     if args.visualize:
    #         do_visualize(MODEL, args.load)
    #     else:
    #         predcfg = PredictConfig(
    #             model=MODEL,
    #             session_init=SmartInit(args.load),
    #             input_names=MODEL.get_inference_tensor_names()[0],
    #             output_names=MODEL.get_inference_tensor_names()[1])

    #         if args.output_pb:
    #             ModelExporter(predcfg).export_compact(args.output_pb, optimize=False)
    #         elif args.output_serving:
    #             ModelExporter(predcfg).export_serving(args.output_serving, optimize=False)

    #         if args.predict:
    #             predictor = OfflinePredictor(predcfg)
    #             for image_file in args.predict:
    #                 do_predict(predictor, image_file)
    #         elif args.evaluate:
    #             assert args.evaluate.endswith('.json'), args.evaluate
    #             do_evaluate(predcfg, args.evaluate)
    #         elif args.benchmark:
    #             df = get_eval_dataflow(cfg.DATA.VAL[0])
    #             df.reset_state()
    #             predictor = OfflinePredictor(predcfg)
    #             for _, img in enumerate(tqdm.tqdm(df, total=len(df), smoothing=0.5)):
    #                 # This includes post-processing time, which is done on CPU and not optimized
    #                 # To exclude it, modify `predict_image`.
    #                 predict_image(img[0], predictor)