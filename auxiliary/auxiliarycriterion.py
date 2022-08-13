from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
from fairseq.criterions import register_criterion
import logging
import torch
import math
from fairseq import metrics, utils
import torch.nn.functional as F

@register_criterion('auxiliarycriterion')
class AuxiliaryCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            mask_loss_weight,
    ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.mask_loss_weight = mask_loss_weight

    def add_args(parser):
        parser.add_argument('--mask-loss-weight', default=0., type=float,
                            help='weight of mask loss')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')



    def forward(self, model, sample, reduce=True, show=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # return super().forward(model, sample, reduce=reduce)
        net_output, net_output_auxiliary = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        src_len = net_output[-1]["mask"][0].size()[-1]
        #mask_ave = net_output[-1]["mask"][0].mean(dim=0).mean(dim=0).mean(dim=-1).sum()
        #gate_ave = net_output[-1]["gate"][0].mean(dim=0).mean(dim=0).sum()

        auxiliary_loss, _ = self.compute_loss(model, net_output_auxiliary, sample, reduce=reduce)
        p_norm = torch.norm(1 - net_output[-1]["mask"][0], p=2) / src_len
        auxiliary_loss_final = auxiliary_loss - self.mask_loss_weight * p_norm






        logging_output = {
            "loss": loss.data,
            "auxiliary_loss": auxiliary_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        del nll_loss, auxiliary_loss, p_norm
        return loss, auxiliary_loss_final, sample_size, logging_output

    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        auxiliary_loss_sum = sum(log.get('auxiliary_loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar('auxiliary_loss', auxiliary_loss_sum / sample_size / math.log(2), sample_size, round=6)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

