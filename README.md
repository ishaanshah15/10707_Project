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



<img width="598" alt="Screen Shot 2023-11-06 at 7 22 08 PM" src="https://github.com/ishaanshah15/10707_Project/assets/114367751/50cdabc6-0b81-46fd-84df-56f0ffa02071">
