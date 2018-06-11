# Learning from small amount of labeled data 
**Few-shot learning** is a general umbrella term for learning models with a small amount of data. 
It is usually assumed that the models are learned in supervised manner, and that the _labeled_ data is scarce. 
The approaches to learning from small amount labeled data are:
* **transfer learning**: learning new model by adapting an existing model, usually learned for related tasks
* **semi-supervised learning**: learning with small amount of labeled data and a larger amount of unlabeled data
* **meta-learning**: learning to learn with small amount of labeled data by learning in few-shot episodes

<details>
<summary>Matching Networks for One Shot Learning, <a href="https://arxiv.org/abs/1606.04080">link</a>, meta-learning</summary>
<p>
Learning from a few examples remains a key challenge in machine learning. Despite recent advances in important domains such as vision and language, the standard supervised deep learning paradigm does not offer a satisfactory solution for learning new concepts rapidly from little data. In this work, we employ ideas from metric learning based on deep neural features and from recent advances that augment neural networks with external memories. <b>Our framework learns a network that maps a small labelled support set and an unlabelled example to its label, obviating the need for fine-tuning to adapt to new class types.</b> We then define one-shot learning problems on vision (using Omniglot, ImageNet) and language tasks. Our algorithm improves one-shot accuracy on ImageNet from 87.6% to 93.2% and from 88.0% to 93.8% on Omniglot compared to competing approaches. We also demonstrate the usefulness of the same model on language modeling by introducing a one-shot task on the Penn Treebank.
</p>
</details>
<details>
<summary>Optimization as a model for few-shot learning<a href="https://openreview.net/pdf?id=rJY0-Kcll">link</a>, meta-learning</summary>
<p>
Though deep neural networks have shown great success in the large data domain, they generally perform poorly on few-shot learning tasks, where a classifier has to quickly generalize after seeing very few examples from each class. The general belief is that gradient-based optimization in high capacity classifiers requires many iterative steps over many examples to perform well. <b>Here, we propose an LSTM-based meta-learner model to learn the exact optimization algorithm used to train another learner neural network classifier in the few-shot regime. The parametrization of our model allows it to learn appropriate parameter updates specifically for the scenario where a set amount of updates will be made, while also learning a general initialization of the learner (classifier) network that allows for quick convergence of training.</b> We demonstrate that this meta-learning model is competitive with deep metric-learning techniques for few-shot learning.
</p>
</details>
<details>
<summary>Prototypical Networks for Few-shot Learning<a href="https://arxiv.org/abs/1703.05175">link</a>, meta-learning</summary>
<p>
We propose prototypical networks for the problem of few-shot classification, where a classifier must generalize to new classes not seen in the training set, given only a small number of examples of each new class. <b>Prototypical networks learn a metric space in which classification can be performed by computing distances to prototype representations of each class.</b> Compared to recent approaches for few-shot learning, they reflect a simpler inductive bias that is beneficial in this limited-data regime, and achieve excellent results. We provide an analysis showing that some simple design decisions can yield substantial improvements over recent approaches involving complicated architectural choices and meta-learning. We further extend prototypical networks to zero-shot learning and achieve state-of-the-art results on the CU-Birds dataset.
</p>
</details>
<details>
<summary>Meta-Learning for Semi-Supervised Few-Shot Classification<a href="https://arxiv.org/abs/1803.00676">link</a>, meta-learning, semi-supervised learning</summary>
<p>
In few-shot classification, we are interested in learning algorithms that train a classifier from only a handful of labeled examples. Recent progress in few-shot classification has featured meta-learning, in which a parameterized model for a learning algorithm is defined and trained on episodes representing different classification problems, each with a small labeled training set and its corresponding test set. <b>In this work, we advance this few-shot classification paradigm towards a scenario where unlabeled examples are also available within each episode. We consider two situations: one where all unlabeled examples are assumed to belong to the same set of classes as the labeled examples of the episode, as well as the more challenging situation where examples from other distractor classes are also provided.</b> To address this paradigm, we propose novel extensions of Prototypical Networks (Snell et al., 2017) that are augmented with the ability to use unlabeled examples when producing prototypes. These models are trained in an end-to-end way on episodes, to learn to leverage the unlabeled examples successfully. We evaluate these methods on versions of the Omniglot and miniImageNet benchmarks, adapted to this new framework augmented with unlabeled examples. We also propose a new split of ImageNet, consisting of a large set of classes, with a hierarchical structure. Our experiments confirm that our Prototypical Networks can learn to improve their predictions due to unlabeled examples, much like a semi-supervised algorithm would.
</p>
</details>
