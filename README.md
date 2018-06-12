# Learning from small amount of labeled data
## Overview
**Few-shot learning** is a general umbrella term for learning models with a small amount of data. 
It is usually assumed that the models are learned in supervised manner, and that the _labeled_ data is scarce. 
The approaches to learning from small amount labeled data are:
* **transfer learning**: learning new model by adapting an existing model, usually learned for related tasks
* **semi-supervised learning**: learning with small amount of labeled data and a larger amount of unlabeled data
* **meta-learning**: learning to learn with small amount of labeled data by learning in few-shot episodes
Related fields:
* **unsupervised learning**: learning a good representation for the data without any labels
* **zero-shot learning**: learning to recognize classes for which labels have not been provided during training

## Literature
<details>
<summary>Matching Networks for One Shot Learning <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1606.04080">paper</a>

---

### Abstract

Learning from a few examples remains a key challenge in machine learning. Despite recent advances in important domains such as vision and language, the standard supervised deep learning paradigm does not offer a satisfactory solution for learning new concepts rapidly from little data. In this work, we employ ideas from metric learning based on deep neural features and from recent advances that augment neural networks with external memories. <b>Our framework learns a network that maps a small labelled support set and an unlabelled example to its label, obviating the need for fine-tuning to adapt to new class types.</b> We then define one-shot learning problems on vision (using Omniglot, ImageNet) and language tasks. Our algorithm improves one-shot accuracy on ImageNet from 87.6% to 93.2% and from 88.0% to 93.8% on Omniglot compared to competing approaches. We also demonstrate the usefulness of the same model on language modeling by introducing a one-shot task on the Penn Treebank.

---
</p>
</details>

<details>
<summary>Optimization as a model for few-shot learning <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://openreview.net/pdf?id=rJY0-Kcll">paper</a> 

---

Though deep neural networks have shown great success in the large data domain, they generally perform poorly on few-shot learning tasks, where a classifier has to quickly generalize after seeing very few examples from each class. The general belief is that gradient-based optimization in high capacity classifiers requires many iterative steps over many examples to perform well. <b>Here, we propose an LSTM-based meta-learner model to learn the exact optimization algorithm used to train another learner neural network classifier in the few-shot regime. The parametrization of our model allows it to learn appropriate parameter updates specifically for the scenario where a set amount of updates will be made, while also learning a general initialization of the learner (classifier) network that allows for quick convergence of training.</b> We demonstrate that this meta-learning model is competitive with deep metric-learning techniques for few-shot learning.

---
</p>
</details>

<details>
<summary>Prototypical Networks for Few-shot Learning <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1703.05175">paper</a> 

---

We propose prototypical networks for the problem of few-shot classification, where a classifier must generalize to new classes not seen in the training set, given only a small number of examples of each new class. <b>Prototypical networks learn a metric space in which classification can be performed by computing distances to prototype representations of each class.</b> Compared to recent approaches for few-shot learning, they reflect a simpler inductive bias that is beneficial in this limited-data regime, and achieve excellent results. We provide an analysis showing that some simple design decisions can yield substantial improvements over recent approaches involving complicated architectural choices and meta-learning. We further extend prototypical networks to zero-shot learning and achieve state-of-the-art results on the CU-Birds dataset.

---
</p>
</details>

<details>
<summary>Meta-Learning for Semi-Supervised Few-Shot Classification <kbd>meta-learning</kbd>, <kbd>semi-supervised</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1803.00676">paper</a>

---

In few-shot classification, we are interested in learning algorithms that train a classifier from only a handful of labeled examples. Recent progress in few-shot classification has featured meta-learning, in which a parameterized model for a learning algorithm is defined and trained on episodes representing different classification problems, each with a small labeled training set and its corresponding test set. <b>In this work, we advance this few-shot classification paradigm towards a scenario where unlabeled examples are also available within each episode. We consider two situations: one where all unlabeled examples are assumed to belong to the same set of classes as the labeled examples of the episode, as well as the more challenging situation where examples from other distractor classes are also provided.</b> To address this paradigm, we propose novel extensions of Prototypical Networks (Snell et al., 2017) that are augmented with the ability to use unlabeled examples when producing prototypes. These models are trained in an end-to-end way on episodes, to learn to leverage the unlabeled examples successfully. We evaluate these methods on versions of the Omniglot and miniImageNet benchmarks, adapted to this new framework augmented with unlabeled examples. We also propose a new split of ImageNet, consisting of a large set of classes, with a hierarchical structure. Our experiments confirm that our Prototypical Networks can learn to improve their predictions due to unlabeled examples, much like a semi-supervised algorithm would.

---
</p>
</details>

<details>
<summary>Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1703.03400">paper</a>, <a href="http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn">blog</a>

---

We propose an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning. The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. <b>In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task. In effect, our method trains the model to be easy to fine-tune.</b> We demonstrate that this approach leads to state-of-the-art performance on two few-shot image classification benchmarks, produces good results on few-shot regression, and accelerates fine-tuning for policy gradient reinforcement learning with neural network policies.

---
</p>
</details>

<details>
<summary>Recasting Gradient-Based Meta-Learning as Hierarchical Bayes <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1801.08930">paper</a>

---

Meta-learning allows an intelligent agent to leverage prior learning episodes as a basis for quickly improving performance on a novel task. Bayesian hierarchical modeling provides a theoretical framework for formalizing meta-learning as inference for a set of parameters that are shared across tasks. Here, we reformulate the model-agnostic meta-learning algorithm (MAML) of Finn et al. (2017) as a method for probabilistic inference in a hierarchical Bayesian model. In contrast to prior methods for meta-learning via hierarchical Bayes, MAML is naturally applicable to complex function approximators through its use of a scalable gradient descent procedure for posterior inference. Furthermore, the identification of MAML as hierarchical Bayes provides a way to understand the algorithm's operation as a meta-learning procedure, as well as an opportunity to make use of computational strategies for efficient inference. We use this opportunity to propose an improvement to the MAML algorithm that makes use of techniques from approximate inference and curvature estimation.

---
</p>
</details>

<details>
<summary>Reptile: a Scalable Meta-learning Algorithm <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1803.02999">paper</a>, <a href="https://blog.openai.com/reptile">blog</a>

---
This paper considers metalearning problems, where there is a distribution of tasks, and we would like to obtain an agent that performs well (i.e., learns quickly) when presented with a previously unseen task sampled from this distribution. <b>We present a remarkably simple metalearning algorithm called Reptile, which learns a parameter initialization that can be fine-tuned quickly on a new task</b>. Reptile works by repeatedly sampling a task, training on it, and moving the initialization towards the trained weights on that task. <b>Unlike MAML, which also learns an initialization, Reptile doesn't require differentiating through the optimization process, making it more suitable for optimization problems where many update steps are required.</b> We show that Reptile performs well on some well-established benchmarks for few-shot classification. We provide some theoretical analysis aimed at understanding why Reptile works.

---
</p>
</details>

<details>
<summary>Meta-Learning with Memory-Augmented Neural Networks <kbd>meta-learning</kbd></summary>
<p>

---

<a href="http://proceedings.mlr.press/v48/santoro16.pdf">paper</a>

---

Despite recent breakthroughs in the applications of deep neural networks, one setting that presents a persistent challenge is that of “one-shot learning.” Traditional gradient-based networks require a lot of data to learn, often through extensive iterative training. When new data is encountered, the models must inefficiently relearn their parameters to adequately incorporate the new information without catastrophic interference. <b>Architectures with augmented memory capacities, such as Neural Turing Machines (NTMs), offer the ability to quickly encode and retrieve new information, and hence can potentially obviate the downsides of conventional models.</b> Here, we demonstrate the ability of a memory-augmented neural network to rapidly assimilate new data, and leverage this data to make accurate predictions after only a few samples. We also introduce a new method for accessing an external memory that focuses on memory content, unlike previous methods that additionally use memory locationbased focusing mechanisms.

---
</p>
</details>

<details>
<summary>Learning to learn by gradient descent by gradient descent <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1606.04474">paper</a>

---

The move from hand-designed features to learned features in machine learning has been wildly successful. In spite of this, optimization algorithms are still designed by hand. <b>In this paper we show how the design of an optimization algorithm can be cast as a learning problem, allowing the algorithm to learn to exploit structure in the problems of interest in an automatic way.</b> Our learned algorithms, implemented by LSTMs, outperform generic, hand-designed competitors on the tasks for which they are trained, and also generalize well to new tasks with similar structure. We demonstrate this on a number of tasks, including simple convex problems, training neural networks, and styling images with neural art.

---
</p>
</details>


<details>
<summary>Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1801.05558">paper</a>

---

Gradient-based meta-learning has been shown to be expressive enough to approximate any learning algorithm. While previous such methods have been successful in meta-learning tasks, they resort to simple gradient descent during meta-testing. <b>Our primary contribution is the <i>MT-net</i>, which enables the meta-learner to learn on each layer's activation space a subspace that the task-specific learner performs gradient descent on.</b> Additionally, a task-specific learner of an <i>MT-net</i> performs gradient descent with respect to a meta-learned distance metric, which warps the activation space to be more sensitive to task identity. We demonstrate that the dimension of this learned subspace reflects the complexity of the task-specific learner's adaptation task, and also that our model is less sensitive to the choice of initial learning rates than previous gradient-based meta-learning methods. Our method achieves state-of-the-art or comparable performance on few-shot classification and regression tasks.

---
</p>
</details>

<details>
<summary>A Simple Neural Attentive Meta-Learner <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://openreview.net/forum?id=B1DmUzWAW">paper</a>

---

Deep neural networks excel in regimes with large amounts of data, but tend to struggle when data is scarce or when they need to adapt quickly to changes in the task. In response, recent work in meta-learning proposes training a meta-learner on a distribution of similar tasks, in the hopes of generalization to novel but related tasks by learning a high-level strategy that captures the essence of the problem it is asked to solve. However, many recent meta-learning approaches are extensively hand-designed, either using architectures specialized to a particular application, or hard-coding algorithmic components that constrain how the meta-learner solves the task. <b>We propose a class of simple and generic meta-learner architectures that use a novel combination of temporal convolutions and soft attention; the former to aggregate information from past experience and the latter to pinpoint specific pieces of information.</b> In the most extensive set of meta-learning experiments to date, we evaluate the resulting Simple Neural AttentIve Learner (or SNAIL) on several heavily-benchmarked tasks. On all tasks, in both supervised and reinforcement learning, SNAIL attains state-of-the-art performance by significant margins.

---
</p>
</details>

<details>
<summary>Differentiable plasticity: training plastic neural networks with backpropagation  <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1804.02464">paper</a>, <a href="https://eng.uber.com/differentiable-plasticity">blog</a>

---

How can we build agents that keep learning from experience, quickly and efficiently, after their initial training? Here we take inspiration from the main mechanism of learning in biological brains: synaptic plasticity, carefully tuned by evolution to produce efficient lifelong learning. We show that plasticity, just like connection weights, can be optimized by gradient descent in large (millions of parameters) recurrent networks with Hebbian plastic connections. First, recurrent plastic networks with more than two million parameters can be trained to memorize and reconstruct sets of novel, high-dimensional 1000+ pixels natural images not seen during training. Crucially, traditional non-plastic recurrent networks fail to solve this task. Furthermore, trained plastic networks can also solve generic meta-learning tasks such as the Omniglot task, with competitive results and little parameter overhead. Finally, in reinforcement learning settings, plastic networks outperform a non-plastic equivalent in a maze exploration task. We conclude that differentiable plasticity may provide a powerful novel approach to the learning-to-learn problem.

---
</p>
</details>

<details>
<summary>Meta-learning with differentiable closed-form solvers <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1805.08136">paper</a>, <a href="http://www.robots.ox.ac.uk/~luca/r2d2.html">project page</a>

---

Adapting deep networks to new concepts from few examples is extremely challenging, due to the high computational and data requirements of standard fine-tuning procedures. Most works on meta-learning and few-shot learning have thus focused on simple learning techniques for adaptation, such as nearest neighbors or gradient descent. Nonetheless, the machine learning literature contains a wealth of methods that learn non-deep models very efficiently. <b>In this work we propose to use these fast convergent methods as the main adaptation mechanism for few-shot learning. The main idea is to teach a deep network to use standard machine learning tools, such as logistic regression, as part of its own internal model, enabling it to quickly adapt to novel tasks.</b> This requires back-propagating errors through the solver steps. While normally the matrix operations involved would be costly, the small number of examples works to our advantage, by making use of the Woodbury identity. We propose both iterative and closed-form solvers, based on logistic regression and ridge regression components. Our methods achieve excellent performance on three few-shot learning benchmarks, showing competitive performance on Omniglot and surpassing all state-of-the-art alternatives on miniImageNet and CIFAR-100.

---
</p>
</details>

<details>
<summary>Piecewise classifier mappings: Learning fine-grained learners for novel categories with few examples <kbd>meta-learning</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1805.04288">paper</a>

---

Humans are capable of learning a new fine-grained concept with very little supervision, e.g., few exemplary images for a species of bird, yet our best deep learning systems need hundreds or thousands of labeled examples. In this paper, we try to reduce this gap by studying the fine-grained image recognition problem in a challenging few-shot learning setting, termed few-shot fine-grained recognition (FSFG). The task of FSFG requires the learning systems to build classifiers for novel fine-grained categories from few examples (only one or less than five). To solve this problem, we propose an end-to-end trainable deep network which is inspired by the state-of-the-art fine-grained recognition model and is tailored for the FSFG task. <b>Specifically, our network consists of a bilinear feature learning module and a classifier mapping module: while the former encodes the discriminative information of an exemplar image into a feature vector, the latter maps the intermediate feature into the decision boundary of the novel category.</b> The key novelty of our model is a "piecewise mappings" function in the classifier mapping module, which generates the decision boundary via learning a set of more attainable sub-classifiers in a more parameter-economic way. We learn the exemplar-to-classifier mapping based on an auxiliary dataset in a meta-learning fashion, which is expected to be able to generalize to novel categories. By conducting comprehensive experiments on three fine-grained datasets, we demonstrate that the proposed method achieves superior performance over the competing baselines.

---
</p>
</details>

<details>
<summary>Learning to Learn: Model Regression Networks for Easy Small Sample Learning <kbd>transfer</kbd></summary>
<p>

---

<a href="https://www.ri.cmu.edu/pub_files/2016/10/yuxiongw_eccv16_learntolearn.pdf">paper</a>

---

We develop a conceptually simple but powerful approach that can learn novel categories from few annotated examples. <b>In this approach, the experience with already learned categories is used to facilitate the learning of novel classes. Our insight is two-fold: 1) there exists a generic, category agnostic transformation from models learned from few samples to models learned from large enough sample sets, and 2) such a transformation could be effectively learned by high-capacity regressors.</b> In particular, we automatically learn the transformation with a deep model regression network on a large collection of model pairs. Experiments demonstrate that encoding this transformation as prior knowledge greatly facilitates the recognition in the small sample size regime on a broad range of tasks, including domain adaptation, fine-grained recognition, action recognition, and scene classification.

---
</p>
</details>

<details>
<summary>Few-Shot Learning Through an Information Retrieval Lens <kbd>meta-learning</kbd>, <kbd>transfer</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1707.02610">paper</a>

---
Few-shot learning refers to understanding new concepts from only a few examples. We propose an information retrieval-inspired approach for this problem that is motivated by the increased importance of maximally leveraging all the available information in this low-data regime. We define a training objective that aims to extract as much information as possible from each training batch by effectively optimizing over all relative orderings of the batch points simultaneously. <b>In particular, we view each batch point as a `query' that ranks the remaining ones based on its predicted relevance to them and we define a model within the framework of structured prediction to optimize mean Average Precision over these rankings.</b> Our method achieves impressive results on the standard few-shot classification benchmarks while is also capable of few-shot retrieval.

---
</p>
</details>

<details>
<summary>Learning from Small Sample Sets by Combining Unsupervised Meta-Training with CNNs <kbd>meta-learning</kbd>, <kbd>semi-supervised</kbd></summary>
<p>

---

<a href="https://www.ri.cmu.edu/wp-content/uploads/2017/06/yuxiongw_nips16_ldscnn.pdf">paper</a>

---

This work explores CNNs for the recognition of novel categories from few examples. Inspired by the transferability properties of CNNs, we introduce an additional unsupervised meta-training stage that exposes multiple top layer units to a large amount of unlabeled real-world images. By encouraging these units to learn diverse sets of low-density separators across the unlabeled data, we capture a more generic, richer description of the visual world, which decouples these units from ties to a specific set of categories. We propose an unsupervised margin maximization that jointly estimates compact high-density regions and infers low-density separators. The low-density separator (LDS) modules can be plugged into any or all of the top layers of a standard CNN architecture. The resulting CNNs significantly improve the performance in scene classification, fine-grained recognition, and action recognition with small training samples.

---
</p>
</details>

<details>
<summary>Growing a Brain: Fine-Tuning by Increasing Model Capacity <kbd>transfer</kbd></summary>
<p>

---

<a href="https://www.ri.cmu.edu/wp-content/uploads/2017/06/yuxiongw_cvpr17_growingcnn.pdf">paper</a>

---

CNNs have made an undeniable impact on computer vision through the ability to learn high-capacity models with large annotated training sets. One of their remarkable properties is the ability to transfer knowledge from a large source dataset to a (typically smaller) target dataset. This is usually accomplished through fine-tuning a fixed-size network on new target data. Indeed, virtually every contemporary visual recognition system makes use of fine-tuning to transfer knowledge from ImageNet. In this work, we analyze what components and parameters change during finetuning, and discover that increasing model capacity allows for more natural model adaptation through fine-tuning. By making an analogy to developmental learning, we demonstrate that “growing” a CNN with additional units, either by widening existing layers or deepening the overall network, significantly outperforms classic fine-tuning approaches. But in order to properly grow a network, we show that newly-added units must be appropriately normalized to allow for a pace of learning that is consistent with existing units. We empirically validate our approach on several benchmark datasets, producing state-of-the-art results.

---
</p>
</details>

<details>
<summary>Beyond Fine Tuning: A Modular Approach to Learning on Small Data <kbd>transfer</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1611.01714">paper</a>

---

In this paper we present a technique to train neural network models on small amounts of data. Current methods for training neural networks on small amounts of rich data typically rely on strategies such as fine-tuning a pre-trained neural network or the use of domain-specific hand-engineered features. Here we take the approach of treating network layers, or entire networks, as modules and combine pre-trained modules with untrained modules, to learn the shift in distributions between data sets. The central impact of using a modular approach comes from adding new representations to a network, as opposed to replacing representations via fine-tuning. Using this technique, we are able surpass results using standard fine-tuning transfer learning approaches, and we are also able to significantly increase performance over such approaches when using smaller amounts of data.

---
</p>
</details>

<details>
<summary>Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights <kbd>transfer</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1801.06519">paper</a>

---

This work presents a method for adapting a single, fixed deep neural network to multiple tasks without affecting performance on already learned tasks. By building upon ideas from network quantization and pruning, we learn binary masks that piggyback on an existing network, or are applied to unmodified weights of that network to provide good performance on a new task. These masks are learned in an end-to-end differentiable fashion, and incur a low overhead of 1 bit per network parameter, per task. Even though the underlying network is fixed, the ability to mask individual weights allows for the learning of a large number of filters. We show performance comparable to dedicated fine-tuned networks for a variety of classification tasks, including those with large domain shifts from the initial task (ImageNet), and a variety of network architectures. Unlike prior work, we do not suffer from catastrophic forgetting or competition between tasks, and our performance is agnostic to task ordering.

---
</p>
</details>


<details>
<summary>Learning without Forgetting <kbd>transfer</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1606.09282">paper</a>

---

When building a unified vision system or gradually adding new capabilities to a system, the usual assumption is that training data for all tasks is always available. However, as the number of tasks grows, storing and retraining on such data becomes infeasible. A new problem arises where we add new capabilities to a Convolutional Neural Network (CNN), but the training data for its existing capabilities are unavailable. We propose our Learning without Forgetting method, which uses only new task data to train the network while preserving the original capabilities. Our method performs favorably compared to commonly used feature extraction and fine-tuning adaption techniques and performs similarly to multitask learning that uses original task data we assume unavailable. A more surprising observation is that Learning without Forgetting may be able to replace fine-tuning with similar old and new task datasets for improved new task performance.

---
</p>
</details>


<details>
<summary>Knowledge Concentration: Learning 100K Object Classifiers in a Single CNN <kbd>transfer</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1711.07607">paper</a>

---

Fine-grained image labels are desirable for many computer vision applications, such as visual search or mobile AI assistant. These applications rely on image classification models that can produce hundreds of thousands (e.g. 100K) of diversified fine-grained image labels on input images. However, training a network at this vocabulary scale is challenging, and suffers from intolerable large model size and slow training speed, which leads to unsatisfying classification performance. A straightforward solution would be training separate expert networks (specialists), with each specialist focusing on learning one specific vertical (e.g. cars, birds...). However, deploying dozens of expert networks in a practical system would significantly increase system complexity and inference latency, and consumes large amounts of computational resources. To address these challenges, we propose a Knowledge Concentration method, which effectively transfers the knowledge from dozens of specialists (multiple teacher networks) into one single model (one student network) to classify 100K object categories. There are three salient aspects in our method: (1) a multi-teacher single-student knowledge distillation framework; (2) a self-paced learning mechanism to allow the student to learn from different teachers at various paces; (3) structurally connected layers to expand the student network capacity with limited extra parameters. We validate our method on OpenImage and a newly collected dataset, Entity-Foto-Tree (EFT), with 100K categories, and show that the proposed model performs significantly better than the baseline generalist model.

---
</p>
</details>


<details>
<summary>Preserving Semantic Relations for Zero-Shot Learning <kbd>zero-shot learning</kbd></summary>
<p>

---

<a href="https://arxiv.org/abs/1803.03049">paper</a>

---

Zero-shot learning has gained popularity due to its potential to scale recognition models without requiring additional training data. This is usually achieved by associating categories with their semantic information like attributes. However, we believe that the potential offered by this paradigm is not yet fully exploited. In this work, we propose to utilize the structure of the space spanned by the attributes using a set of relations. We devise objective functions to preserve these relations in the embedding space, thereby inducing semanticity to the embedding space. Through extensive experimental evaluation on five benchmark datasets, we demonstrate that inducing semanticity to the embedding space is beneficial for zero-shot learning. The proposed approach outperforms the state-of-the-art on the standard zero-shot setting as well as the more realistic generalized zero-shot setting. We also demonstrate how the proposed approach can be useful for making approximate semantic inferences about an image belonging to a category for which attribute information is not available.

---
</p>
</details>

