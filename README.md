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
<summary>Paper Title 2, Paper Authors 2, Paper Page 2</summary>
<p>This is abstract sample 2</p>
</details>
