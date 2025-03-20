# Benchmark Datasets for Image Classification & Clustering  &  Face Recognition 

A summary and overview of the basic datasets used in computer vision research have been compiled and classified according to their main application fields.

------

## Ⅰ. Image Classification Datasets

### 1. ‌**ImageNet (ILSVRC)**‌

- ‌**Core Concept**‌: Large-scale hierarchical ontology (WordNet) with 14M+ images across 21,841 synsets

- ‌

  Key Features

  ‌:

  - 1,000 object categories for standard classification benchmark
  - High inter-class similarity (e.g., 120 dog breeds)
  - Spatial pyramid pooling requirements

- ‌**Theoretical Significance**‌: Established deep learning supremacy through AlexNet (2012)

### 2. ‌**COCO (Common Objects in Context)**‌

- ‌**Conceptual Framework**‌: Multi-object understanding with rich contextual relationships

- ‌

  Key Properties

  ‌:

  - 330K images with 80 object categories
  - Instance segmentation annotations
  - Crowd-sourcing challenges in object separation

- ‌**Research Impact**‌: Benchmark for contextual reasoning models

### 3. ‌**Places365**‌

- ‌**Theoretical Basis**‌: Scene-centric classification with environmental semantics

- ‌

  Characteristics

  ‌:

  - 1.8M images from 365 scene categories
  - High intra-class variance (e.g., "kitchen" layouts)
  - Spatial layout analysis requirements

- ‌**Application**‌: Environment-aware vision systems

### 4. ‌**CIFAR-10/100**‌

- ‌

  Fundamental Design

  ‌:

  - 60K 32x32 images (10/100 classes)
  - Low-resolution challenge
  - Curriculum learning benchmark

- ‌**Theoretical Role**‌: Baseline for sample efficiency studies

### 5. ‌**FGVC-Aircraft**‌

- ‌**Specialization**‌: Fine-grained visual categorization

- ‌

  Attributes

  ‌:

  - 10,200 images of 100 aircraft variants
  - Manufacturer/model/year differentiation
  - Minimal inter-class variance

- ‌**Research Value**‌: Metric learning testbed

### 6. ‌**Food-101**‌

- ‌**Concept**‌: Culinary domain classification

- ‌

  Properties

  ‌:

  - 101,000 food images
  - High visual similarity (e.g., burgers vs. sandwiches)
  - Cross-cultural annotation challenges

- ‌**Theoretical Interest**‌: Domain adaptation studies

### 7. ‌**SVHN (Street View House Numbers)**‌

- ‌**Core Principle**‌: Real-world digit recognition

- ‌

  Characteristics

  ‌:

  - 600K digit patches from Google Street View
  - Natural distortions and occlusions
  - Multi-digit sequence recognition

- ‌**Significance**‌: Bridge between MNIST and real-world applications

### 8. ‌**CUDS**‌

- ‌**Theoretical Framework**‌: Multi-task learning benchmark

- ‌

  Features

  ‌:

  - 20 object categories,Multiple scenarios and multiple categories
  - Simultaneous classification/detection/segmentation
  - Small dataset challenges (11,530 images)

- ‌**Historical Impact**‌: Defined modern object recognition metrics

### 9. ‌**iNaturalist**‌

- ‌**Ecological Focus**‌: Long-tailed species distribution

- ‌

  Attributes

  ‌:

  - 2.7M images across 10,000+ species
  - Expert-curated taxonomic hierarchy
  - Severe class imbalance

- ‌**Research Utility**‌: Few-shot learning benchmark

### 10. ‌**Gam**‌

- ‌**Conceptual Design**‌: Cross-domain generalization

- ‌

  Properties

  ‌:

  - 6 fields, different scenarios
  - 345 classes with domain shifts
  - 600K+ images

- ‌**Theoretical Value**‌: Domain gap quantification

------

## Ⅱ. Image Clustering Datasets

### 1. ‌**MNIST**‌

- ‌**Clustering Basis**‌: Well-separated manifolds

- ‌

  Cluster Characteristics

  ‌:

  - 70K 28x28 handwritten digits
  - 10 natural clusters
  - Overlap in stylized digits (7 vs 9)

### 2. ‌**COIL-20/100**‌

- ‌**Theoretical Construct**‌: Multi-view object analysis

- ‌

  Properties

  ‌:

  - 1,440 images (20 objects × 72 angles)
  - Continuous pose manifold structure
  - Photometric variations

### 3. ‌**Fashion-MNIST**‌

- ‌

  Cluster Dynamics

  ‌:

  - 10 apparel categories
  - Semantic vs visual cluster discrepancy
  - 70K 28x28 grayscale images

- ‌**Research Use**‌: Cluster purity analysis

### 4. ‌**STL-10**‌

- ‌

  Design Philosophy

  ‌:

  - 100K unlabeled + 13K labeled images
  - 10 predefined classes
  - 96x96 resolution

- ‌**Theoretical Value**‌: Semi-supervised clustering benchmark

### 5. ‌**UMist Faces**‌

- ‌

  Cluster Structure

  ‌:

  - 575 face sequences
  - 20 individuals
  - Non-linear pose manifold

- ‌**Significance**‌: Temporal continuity in clustering

### 6. ‌**Reuters-21578**‌

- ‌

  Multimodal Aspect

  ‌:

  - Text-image pairs
  - 5 document categories
  - Feature heterogeneity challenge

- ‌**Theoretical Interest**‌: Cross-modal clustering

### 7. ‌Fast-clu‌

- ‌

  Cluster Complexity

  ‌:

  - 165 images of 15 subjects
  - Extreme lighting variations

- ‌**Research Challenge**‌: Invariant representation learning

### 8. ‌**COIL-100**‌

- ‌

  Advanced Version

  ‌:

  - 100 objects × 72 views
  - Color variations
  - 3D rotation manifolds

- ‌**Cluster Property**‌: High-dimensional sparsity

### 9. ‌**spss**‌

- ‌

  Hybrid Design

  ‌:

  - Combined MNIST + Fashion-MNIST
  - 100 pseudo-classes
  - Cluster separability analysis

- ‌**Theoretical Use**‌: Cluster robustness testing

### 10. ‌**Synthetic Control Charts**‌

- ‌

  Temporal Clustering

  ‌:

  - 600 control chart time series
  - 6 pattern types
  - Variable-length sequences

- ‌**Significance**‌: Spatio-temporal clustering

------

## Ⅲ. Face Recognition Datasets

------

### 1. ‌**LFW (Labeled Faces in the Wild)**‌

- ‌**Core Concept**‌: Unconstrained face verification benchmark under real-world variations

- ‌

  Key Features

  ‌:

  - 13,233 images of 5,749 identities
  - Extreme pose/lighting/occlusion variations
  - 6,000+ verification pairs for metric learning

- ‌

  Theoretical Significance

  ‌:

  - Established benchmarks for open-set recognition
  - Foundation for deep metric loss studies (e.g., Contrastive, Triplet Loss)

------

### 2. ‌**VGGFace2**‌

- ‌**Core Concept**‌: Large-scale identity recognition with age/pose diversity

- ‌

  Key Features

  ‌:

  - 3.31M images of 9,131 subjects
  - Average 362 images per identity
  - Annotated with age, pose, and ethnicity metadata

- ‌

  Theoretical Significance

  ‌:

  - Supports cross-age face recognition research
  - Benchmark for identity-preserving representation learning

------

### 3. ‌**MS-Celeb-1M**‌

- ‌**Core Concept**‌: Million-scale noisy web-crawled recognition challenge

- ‌

  Key Features

  ‌:

  - 10M images of 100K celebrities
  - Label noise mitigation requirements
  - Long-tailed identity distribution

- ‌

  Theoretical Significance

  ‌:

  - Tests robustness to real-world data imperfections
  - Pre-training standard for industrial-scale systems

------

### 4. ‌**MegaFace**‌

- ‌**Core Concept**‌: Extreme-scale recognition with 1M+ distractors

- ‌

  Key Features

  ‌:

  - Gallery: 690K images of 1,067 identities
  - Probe: 3,530 identity templates
  - 1M+ "distractor" images for FP rate analysis

- ‌

  Theoretical Significance

  ‌:

  - Quantifies recognition scalability limits
  - Evaluates false positive robustness in large galleries

------

### 5. ‌**IJB-C (NIST IARPA Janus Benchmark – C)**‌

- ‌**Core Concept**‌: Template-based recognition with multimodal fusion

- ‌

  Key Features

  ‌:

  - 21,294 stills + 11,142 video frames of 3,531 subjects
  - Composite templates (multiple media per identity)
  - Cross-sensor (visible/thermal) evaluation

- ‌

  Theoretical Significance

  ‌:

  - Models real-world identity aggregation scenarios
  - Benchmark for heterogeneous data fusion methods

### 6. ‌**MSrface**‌

- ‌**Theoretical Role**‌: Pretraining benchmark for deep metric learning

- ‌

  Characteristics

  ‌:

  - 4984 images of 575 identities
  - Semi-automated cleaning (noise rate <5%)
  - Cross-resolution/quality variations

- ‌

  Value

  ‌:

  - Explores hierarchical network depth vs. identity separability
  - Standardized testing platform for triplet loss variants

------

### 7. ‌**CelebA**‌

- ‌**Theoretical Dimension**‌: Entanglement of identity and attributes (expressions/accessories)

- ‌

  Data Features

  ‌:

  - 202,599 images of 10,177 identities
  - 40 binary attribute annotations (glasses/hats/smiles, etc.)
  - Adversarial sample generation across attribute combinations

- ‌

  Significance

  ‌:

  - Quantifies attribute-induced identity confidence shifts
  - Supports causal reasoning in disentangled representations

------

### 8. ‌**YouTube Faces DB**‌

- ‌**Theoretical Focus**‌: Temporal dynamic feature extraction

- ‌

  Characteristics

  ‌:

  - 3,425 video clips of 1,595 identities
  - Average 2.15 clips per identity
  - Unedited real-world motion blur/frame drops

- ‌

  Contribution

  ‌:

  - Demonstrates temporal aggregation superiority over single-frame recognition
  - Drives joint optimization theories for 3D CNNs and LSTMs

------

### 9. ‌**DSO**‌

- ‌**Scientific Question**‌: Geometric consistency across extreme poses

- ‌

  Data

  ‌:

  - 7,000 frontal-profile pairs of 500 identities
  - Controlled pose deviations (±90°)
  - High-precision 3D landmark annotations

- ‌

  Significance

  ‌:

  - Validates polar coordinate mapping effectiveness
  - Defines mathematical boundaries for cross-pose recognizability

------

### 10. ‌**Tagetman Database**‌

- ‌**Theoretical Challenge**‌: Joint robustness to occlusion and illumination

- ‌

  Characteristics

  ‌:

  - 4,00+ images of 100 identities
  - Refer to LFW
  - Systematically controlled lighting direction/intensity
  - Artificial occlusions (glasses/scarves)

- ‌

  Contribution

  ‌:

  - Proposes local attention mechanisms for feature recovery
  - Validates physical models for illumination normalization



These data have been stored on cloud disks, you need to send them to derekkarshuang@gmail.com
