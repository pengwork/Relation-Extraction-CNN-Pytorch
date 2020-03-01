import argparse
import logging
import os
import torch
from sklearn.metrics import precision_recall_fscore_support

from data import Dataset, BatchWrapper
from model.net import Net
import utils.tool as tool

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/', help="Directory containing the dataset")
parser.add_argument('--embedding_pkl_path', default='./data/word_embedding', help="Path to word vecfile.")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--bert', default=False, help=" use Bert or wordembedding")
parser.add_argument('--gpu', default=True, help="use GPU")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")


def evaluate(model, test_data, metric_labels):
    """Evaluate the model on `num_steps` batches."""
    # set model to evaluation mode
    model.eval()

    loss_avg = tool.RunningAverage()
    output_labels = list()
    target_labels = list()

    # compute metrics over the dataset
    for i, batch_data in enumerate(test_data):
        # fetch the next evaluation batch
        words, pos1s, lens, pos2s, labels = batch_data
        # compute model output
        outputs = model(words, pos1s, pos2s)

        loss = model.loss(outputs, labels)
        loss_avg.update(loss.cpu().item())

        batch_output_labels = torch.max(outputs, dim=1)[1]
        output_labels.extend(batch_output_labels.data.cpu().numpy().tolist())
        target_labels.extend(labels.data.cpu().numpy().tolist())

    # Calculate precision, recall and F1 for all relation categories
    p_r_f1_s = precision_recall_fscore_support(target_labels, output_labels, labels=metric_labels, average='micro')
    p_r_f1 = {'precison': p_r_f1_s[0] * 100,
              'recall': p_r_f1_s[1] * 100,
              'f1': p_r_f1_s[2] * 100,
              'loss': loss_avg()}
    return p_r_f1


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = tool.Params(json_path)

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)

    # Get the logger
    tool.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    data_loader = Dataset(args=args, params=params)
    val_data = BatchWrapper(data_loader.get_data('test'), args.gpu)
    metric_labels = list(range(1, 19))  # relation labels to be evaluated
    logging.info("- done.")

    # Define the model and optimizer
    model = Net(args, params)

    logging.info("Starting evaluation...")
    # Reload weights from the saved file
    tool.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, val_data, metric_labels)

    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test metrics: " + metrics_str)
