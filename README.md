# BirdCLEF 2024
## Overview
Birds are excellent indicators of biodiversity change since they are highly mobile and have diverse habitat requirements. Changes in species assemblage and the number of birds can thus indicate the success or failure of a restoration project. However, frequently conducting traditional observer-based bird biodiversity surveys over large areas is expensive and logistically challenging. In comparison, passive acoustic monitoring (PAM) combined with new analytical tools based on machine learning allows conservationists to sample much greater spatial scales with higher temporal resolution and explore the relationship between restoration interventions and biodiversity in depth.
## Goal
For this competition, you'll use your machine-learning skills to identify under-studied **Indian bird species by sound**. Specifically, you'll develop computational solutions to process continuous audio data and recognize the species by their calls. The best entries will be able to train reliable classifiers with limited training data. If successful, you'll help advance ongoing efforts to protect avian biodiversity in the Western Ghats, India, including those led by V. V. Robin's Lab at IISER Tirupati.
## EDA
### Summary of the Train  MetaData
![img](/assets/summary.png)

### Summary of the Train MetaData After Processing
![img](/assets/summary2.png)
- As we can see most of the Missing values came from Secondary labels and type features
- Top Primary label is `zitcis1`

### Over Categorical Features
#### Top 25 Analysis
![img](/assets/primary-label-CatPlot.svg)
![img](/assets/type-CatPlot.svg)

### Distribution of Birds By Location
![img](/assets/Location%20Distribution.png)
bird species of the sky-islands of the Western Ghats in soundscape data.
![img](/assets/newplot%20(7).png)