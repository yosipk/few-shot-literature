# Learning from small amount of labeled data
## Overview
**Few-shot learning** is a general umbrella term for learning models with a small amount of data. 
It is usually assumed that the models are learned in supervised manner, and that the _labeled_ data is scarce. 
The approaches to learning from small amount labeled data are:
* **transfer learning**: learning new model by adapting an existing model, usually learned for related tasks
* **semi-supervised learning**: learning with small amount of labeled data and a larger amount of unlabeled data
* **meta-learning**: learning to learn with small amount of labeled data by learning in few-shot episodes

## Literature
<details>
<summary>Matching Networks for One Shot Learning, <a href="https://arxiv.org/abs/1606.04080">paper</a>, <img src="meta-learning.png" align="top"></summary>
<p>
Learning from a few examples remains a key challenge in machine learning. Despite recent advances in important domains such as vision and language, the standard supervised deep learning paradigm does not offer a satisfactory solution for learning new concepts rapidly from little data. In this work, we employ ideas from metric learning based on deep neural features and from recent advances that augment neural networks with external memories. <b>Our framework learns a network that maps a small labelled support set and an unlabelled example to its label, obviating the need for fine-tuning to adapt to new class types.</b> We then define one-shot learning problems on vision (using Omniglot, ImageNet) and language tasks. Our algorithm improves one-shot accuracy on ImageNet from 87.6% to 93.2% and from 88.0% to 93.8% on Omniglot compared to competing approaches. We also demonstrate the usefulness of the same model on language modeling by introducing a one-shot task on the Penn Treebank.
</p>
</details>
<details>
<summary>Optimization as a model for few-shot learning, <a href="https://openreview.net/pdf?id=rJY0-Kcll">paper</a>, meta-learning</summary>
<p>
Though deep neural networks have shown great success in the large data domain, they generally perform poorly on few-shot learning tasks, where a classifier has to quickly generalize after seeing very few examples from each class. The general belief is that gradient-based optimization in high capacity classifiers requires many iterative steps over many examples to perform well. <b>Here, we propose an LSTM-based meta-learner model to learn the exact optimization algorithm used to train another learner neural network classifier in the few-shot regime. The parametrization of our model allows it to learn appropriate parameter updates specifically for the scenario where a set amount of updates will be made, while also learning a general initialization of the learner (classifier) network that allows for quick convergence of training.</b> We demonstrate that this meta-learning model is competitive with deep metric-learning techniques for few-shot learning.
</p>
</details>
<details>
<summary>Prototypical Networks for Few-shot Learning, <a href="https://arxiv.org/abs/1703.05175">paper</a>, meta-learning</summary>
<p>
We propose prototypical networks for the problem of few-shot classification, where a classifier must generalize to new classes not seen in the training set, given only a small number of examples of each new class. <b>Prototypical networks learn a metric space in which classification can be performed by computing distances to prototype representations of each class.</b> Compared to recent approaches for few-shot learning, they reflect a simpler inductive bias that is beneficial in this limited-data regime, and achieve excellent results. We provide an analysis showing that some simple design decisions can yield substantial improvements over recent approaches involving complicated architectural choices and meta-learning. We further extend prototypical networks to zero-shot learning and achieve state-of-the-art results on the CU-Birds dataset.
</p>
</details>
<details>
<summary>Meta-Learning for Semi-Supervised Few-Shot Classification, <a href="https://arxiv.org/abs/1803.00676">paper</a>, meta-learning, semi-supervised learning</summary>
<p>
In few-shot classification, we are interested in learning algorithms that train a classifier from only a handful of labeled examples. Recent progress in few-shot classification has featured meta-learning, in which a parameterized model for a learning algorithm is defined and trained on episodes representing different classification problems, each with a small labeled training set and its corresponding test set. <b>In this work, we advance this few-shot classification paradigm towards a scenario where unlabeled examples are also available within each episode. We consider two situations: one where all unlabeled examples are assumed to belong to the same set of classes as the labeled examples of the episode, as well as the more challenging situation where examples from other distractor classes are also provided.</b> To address this paradigm, we propose novel extensions of Prototypical Networks (Snell et al., 2017) that are augmented with the ability to use unlabeled examples when producing prototypes. These models are trained in an end-to-end way on episodes, to learn to leverage the unlabeled examples successfully. We evaluate these methods on versions of the Omniglot and miniImageNet benchmarks, adapted to this new framework augmented with unlabeled examples. We also propose a new split of ImageNet, consisting of a large set of classes, with a hierarchical structure. Our experiments confirm that our Prototypical Networks can learn to improve their predictions due to unlabeled examples, much like a semi-supervised algorithm would.
</p>
</details>
<details>
<summary>Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks, <a href="https://arxiv.org/abs/1703.03400">paper</a>, <a href="http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn">blog</a>, meta-learning</summary>
<p>
We propose an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning. The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. <b>In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task. In effect, our method trains the model to be easy to fine-tune.</b> We demonstrate that this approach leads to state-of-the-art performance on two few-shot image classification benchmarks, produces good results on few-shot regression, and accelerates fine-tuning for policy gradient reinforcement learning with neural network policies.
</p>
</details>
<details>
<summary>Recasting Gradient-Based Meta-Learning as Hierarchical Bayes, <a href="https://arxiv.org/abs/1801.08930">paper</a>, meta-learning</summary>
<p>
Meta-learning allows an intelligent agent to leverage prior learning episodes as a basis for quickly improving performance on a novel task. Bayesian hierarchical modeling provides a theoretical framework for formalizing meta-learning as inference for a set of parameters that are shared across tasks. Here, we reformulate the model-agnostic meta-learning algorithm (MAML) of Finn et al. (2017) as a method for probabilistic inference in a hierarchical Bayesian model. In contrast to prior methods for meta-learning via hierarchical Bayes, MAML is naturally applicable to complex function approximators through its use of a scalable gradient descent procedure for posterior inference. Furthermore, the identification of MAML as hierarchical Bayes provides a way to understand the algorithm's operation as a meta-learning procedure, as well as an opportunity to make use of computational strategies for efficient inference. We use this opportunity to propose an improvement to the MAML algorithm that makes use of techniques from approximate inference and curvature estimation.
</p>
</details>
<details>
<summary>Reptile: a Scalable Meta-learning Algorithm, <a href="https://arxiv.org/abs/1803.02999">paper</a>, <a href="https://blog.openai.com/reptile">blog</a>, meta-learning</summary>
<p>
This paper considers metalearning problems, where there is a distribution of tasks, and we would like to obtain an agent that performs well (i.e., learns quickly) when presented with a previously unseen task sampled from this distribution. <b>We present a remarkably simple metalearning algorithm called Reptile, which learns a parameter initialization that can be fine-tuned quickly on a new task</b>. Reptile works by repeatedly sampling a task, training on it, and moving the initialization towards the trained weights on that task. <b>Unlike MAML, which also learns an initialization, Reptile doesn't require differentiating through the optimization process, making it more suitable for optimization problems where many update steps are required.</b> We show that Reptile performs well on some well-established benchmarks for few-shot classification. We provide some theoretical analysis aimed at understanding why Reptile works.
</p>
</details>
<details>
<summary>Meta-Learning with Memory-Augmented Neural Networks, <a href="http://proceedings.mlr.press/v48/santoro16.pdf">paper</a>, meta-learning</summary>
<p>
Despite recent breakthroughs in the applications of deep neural networks, one setting that presents a persistent challenge is that of “one-shot learning.” Traditional gradient-based networks require a lot of data to learn, often through extensive iterative training. When new data is encountered, the models must inefficiently relearn their parameters to adequately incorporate the new information without catastrophic interference. <b>Architectures with augmented memory capacities, such as Neural Turing Machines (NTMs), offer the ability to quickly encode and retrieve new information, and hence can potentially obviate the downsides of conventional models.</b> Here, we demonstrate the ability of a memory-augmented neural network to rapidly assimilate new data, and leverage this data to make accurate predictions after only a few samples. We also introduce a new method for accessing an external memory that focuses on memory content, unlike previous methods that additionally use memory locationbased focusing mechanisms.
</p>
</details>
<details>
<summary>Learning to learn by gradient descent by gradient descent, <a href="https://arxiv.org/abs/1606.04474">paper</a>, meta-learning</summary>
<p>
The move from hand-designed features to learned features in machine learning has been wildly successful. In spite of this, optimization algorithms are still designed by hand. <b>In this paper we show how the design of an optimization algorithm can be cast as a learning problem, allowing the algorithm to learn to exploit structure in the problems of interest in an automatic way.</b> Our learned algorithms, implemented by LSTMs, outperform generic, hand-designed competitors on the tasks for which they are trained, and also generalize well to new tasks with similar structure. We demonstrate this on a number of tasks, including simple convex problems, training neural networks, and styling images with neural art.
</p>
</details>
<details>
<summary>Learning to Learn: Model Regression Networks for Easy Small Sample Learning, <a href="https://www.ri.cmu.edu/pub_files/2016/10/yuxiongw_eccv16_learntolearn.pdf">paper</a>, transfer learning</summary>
<p>
We develop a conceptually simple but powerful approach that can learn novel categories from few annotated examples. <b>In this approach, the experience with already learned categories is used to facilitate the learning of novel classes. Our insight is two-fold: 1) there exists a generic, category agnostic transformation from models learned from few samples to models learned from large enough sample sets, and 2) such a transformation could be effectively learned by high-capacity regressors.</b> In particular, we automatically learn the transformation with a deep model regression network on a large collection of model pairs. Experiments demonstrate that encoding this transformation as prior knowledge greatly facilitates the recognition in the small sample size regime on a broad range of tasks, including domain adaptation, fine-grained recognition, action recognition, and scene classification.
</p>
</details>
<details>
<summary>Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace, <a href="https://arxiv.org/abs/1801.05558">paper</a>, meta-learning</summary>
<p>
Gradient-based meta-learning has been shown to be expressive enough to approximate any learning algorithm. While previous such methods have been successful in meta-learning tasks, they resort to simple gradient descent during meta-testing. <b>Our primary contribution is the <i>MT-net</i>, which enables the meta-learner to learn on each layer's activation space a subspace that the task-specific learner performs gradient descent on.</b> Additionally, a task-specific learner of an <i>MT-net</i> performs gradient descent with respect to a meta-learned distance metric, which warps the activation space to be more sensitive to task identity. We demonstrate that the dimension of this learned subspace reflects the complexity of the task-specific learner's adaptation task, and also that our model is less sensitive to the choice of initial learning rates than previous gradient-based meta-learning methods. Our method achieves state-of-the-art or comparable performance on few-shot classification and regression tasks.
</p>
</details>
<details>
<summary>A Simple Neural Attentive Meta-Learner,<a href="https://openreview.net/forum?id=B1DmUzWAW">paper</a>, meta-learning</summary>
<p>
Deep neural networks excel in regimes with large amounts of data, but tend to struggle when data is scarce or when they need to adapt quickly to changes in the task. In response, recent work in meta-learning proposes training a meta-learner on a distribution of similar tasks, in the hopes of generalization to novel but related tasks by learning a high-level strategy that captures the essence of the problem it is asked to solve. However, many recent meta-learning approaches are extensively hand-designed, either using architectures specialized to a particular application, or hard-coding algorithmic components that constrain how the meta-learner solves the task. <b>We propose a class of simple and generic meta-learner architectures that use a novel combination of temporal convolutions and soft attention; the former to aggregate information from past experience and the latter to pinpoint specific pieces of information.</b> In the most extensive set of meta-learning experiments to date, we evaluate the resulting Simple Neural AttentIve Learner (or SNAIL) on several heavily-benchmarked tasks. On all tasks, in both supervised and reinforcement learning, SNAIL attains state-of-the-art performance by significant margins.
</p>
</details>
<details>
<summary>Differentiable plasticity: training plastic neural networks with backpropagation, <a href="https://arxiv.org/abs/1804.02464">paper</a>, <a href="https://eng.uber.com/differentiable-plasticity">blog</a>, meta-learning</summary>
<p>
How can we build agents that keep learning from experience, quickly and efficiently, after their initial training? Here we take inspiration from the main mechanism of learning in biological brains: synaptic plasticity, carefully tuned by evolution to produce efficient lifelong learning. We show that plasticity, just like connection weights, can be optimized by gradient descent in large (millions of parameters) recurrent networks with Hebbian plastic connections. First, recurrent plastic networks with more than two million parameters can be trained to memorize and reconstruct sets of novel, high-dimensional 1000+ pixels natural images not seen during training. Crucially, traditional non-plastic recurrent networks fail to solve this task. Furthermore, trained plastic networks can also solve generic meta-learning tasks such as the Omniglot task, with competitive results and little parameter overhead. Finally, in reinforcement learning settings, plastic networks outperform a non-plastic equivalent in a maze exploration task. We conclude that differentiable plasticity may provide a powerful novel approach to the learning-to-learn problem.
</p>
</details>
<details>
<summary>Meta-learning with differentiable closed-form solvers, <a href="https://arxiv.org/abs/1805.08136">paper</a>, <a href="http://www.robots.ox.ac.uk/~luca/r2d2.html">project page</a>, meta-learning</summary>
<p>
Adapting deep networks to new concepts from few examples is extremely challenging, due to the high computational and data requirements of standard fine-tuning procedures. Most works on meta-learning and few-shot learning have thus focused on simple learning techniques for adaptation, such as nearest neighbors or gradient descent. Nonetheless, the machine learning literature contains a wealth of methods that learn non-deep models very efficiently. <b>In this work we propose to use these fast convergent methods as the main adaptation mechanism for few-shot learning. The main idea is to teach a deep network to use standard machine learning tools, such as logistic regression, as part of its own internal model, enabling it to quickly adapt to novel tasks.</b> This requires back-propagating errors through the solver steps. While normally the matrix operations involved would be costly, the small number of examples works to our advantage, by making use of the Woodbury identity. We propose both iterative and closed-form solvers, based on logistic regression and ridge regression components. Our methods achieve excellent performance on three few-shot learning benchmarks, showing competitive performance on Omniglot and surpassing all state-of-the-art alternatives on miniImageNet and CIFAR-100.
</p>
</details>
