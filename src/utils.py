import torch
from torch.nn.utils.rnn import pad_sequence


def remove_padding_data(x):
    valid_data_mask = ~torch.isnan(x)
    not_nan_data = torch.masked_select(x, valid_data_mask)
    return not_nan_data


def remove_padding_batch_data(x):
    batch, feat, sequence = x.shape
    valid_data_mask = ~torch.isnan(x)
    not_nan_data = torch.masked_select(x, valid_data_mask)
    return not_nan_data.reshape(batch, feat, -1)


def add_padding_batch_data(outputs):
    batch_size = outputs[0].size(0)
    padded_outputs = []
    for batch in range(batch_size):
        batch_outputs = [o[batch].transpose(0, 1) for o in outputs]
        # pad_sequence はシーケンス長が最初の次元である必要がある
        padded_batch = pad_sequence(batch_outputs, batch_first=True, padding_value=float('nan'))
        # pad_sequenceの結果を再び転置し, チャンネル数を最初の次元に戻す
        padded_batch = padded_batch.transpose(1, 2)
        padded_outputs.append(padded_batch)

    padding_out = torch.stack(padded_outputs)
    return padding_out
