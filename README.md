# 10-707-Course-Project
10-707-Course-Project: Improving Worst-Group Performance via Curriculum Learning and Subnetwork Selection


Team Members: Atharva Kulkarni (CMU LTI) and Ishaan Shah (CMU MLD)

Abstract:
Real-world datasets often exhibit subpopulation shifts, where certain sub-groups
may have different underlying distributions. To effectively train machine learning
models on such datasets, it is important to ensure good performance on each subgroup. However, the standard training paradigm of Empirical Risk Minimization
(ERM) can lead to representation disparities, as it optimizes models by minimizing
the average training loss. Thus, the underrepresented subgroups contribute very little to the overall loss and may therefore incur high error, particularly in the presence
of spurious correlations between inputs, sub-groups, and labels. Therefore, in this
study, we propose a novel approach to address worst-group performance improvement and sub-population shifts by framing them as a continual and curriculum learning problem. Specifically, we opt for the Winning SubNetworks (WSN) technique,
which identifies the most effective subnetworks for each sub-group while preserving
previously chosen weights. Additionally, we presents analysis on three curricula
for determining the sequential learning order. The empirical results indicate that the
different variations of WSN outperform both ERM and groupDRO in terms of performance. Overall, this approach offers a promising solution to the challenge of continual learning and improving worst-group performance.

<img width="493" alt="Screen Shot 2023-11-06 at 7 22 57 PM" src="https://github.com/ishaanshah15/10707_Project/assets/114367751/1d59fc25-6cce-4185-87ea-11f3fcc24940">

Baseline results: Upon analyzing Table 2, it is evident that ERM performs poorly in terms of
accuracy, averaging across the top 5 worst-performing sub-groups. This outcome aligns with ERM’s
approach of training a model by averaging loss over all the datapoints without penalizing the
underperforming sub-groups. On the other hand, groupDRO, though having lower average accuracy
than ERM, yields the highest accuracy on the most challenging sub-group, i.e., ‘people.’ Additionally,
it also outperforms ERM in terms of accuracy on other sub-groups. As groupDRO minimizes the
maximum group loss, it directly optimizes for the worst-group. Hence, it is bound the give good
results. However, due to this strategy, it leads to slight performance drop in the average scores.
Winning Subnetwork results: WSN reports considerably high average accuracy than ERM and
groupDRO, attesting its efficacy in a general setup. While the WSN approach does not improve on the
single worst-group accuracy relative to ERM and GroupDRO, it shows significantly gains when we
consider the five worst subgroups. The subgroup ‘people’ appears to be an outlier and requires more
fine-grained discrimination compared to the other subgroups since the model must attend to relatively
indistinct attributes such as hair, wrinkles, clothes, etc. On the remaining four worst subgroups, the
WSN model achieves more than a 10 percent increase in accuracy compared to GroupDRO and ERM.
It appears that the continual learning formulation and the ability of WSN framework to devote a
percentage of the model weights to each subgroup leads to an overall improvement in the worst-group
performance. One of the other interesting findings is that the average accuracy across also subgroups
also increases compared to ERM. We believe that this could be because there is a mismatch between
maximum likelihood and accuracy, where some outlier images may lead to large cross entropy loss
but optimizing for these outliers may actually lead to lower overall accuracy. We believe that the
superior avg accuracy of WSN over ERM may not hold up when we have a very large dataset, since
the loss would be more stable in those settings. As for the curriculum learning, we find that the choice
of curriculum does not appear to have any perceptible impact on the worst group performance. We
now analyse some of the shortcomings of our approach, which will help us inform future directions
of research.


<img width="598" alt="Screen Shot 2023-11-06 at 7 22 08 PM" src="https://github.com/ishaanshah15/10707_Project/assets/114367751/50cdabc6-0b81-46fd-84df-56f0ffa02071">
