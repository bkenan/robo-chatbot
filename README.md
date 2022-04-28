# Deep Learning ChatBot Microservice for Robo-Advisor

## Background

There are many ways to improve communication between Robo-Advisor and its customers. One effective method in terms of efficiency, effectiveness and results would be to use chatbots.
Several reasons include the following:

1. Customers Prefer Texting to long phone calls
2. It's fast
3. It's available 24 hours

This application has been designed with Python, JavaScript, HTML/CSS, Deep Learning model - BERT and Flask. The idea was to build a chatbot for my Robo-Advisor, so that I can easily integrate it into the web application later. This chatbot receives texts from users and returns intelligent answers based on the trained model.

## Modeling

As for the modeling, I used the BERT (Bidirectional Encoder Representations from Transformers) model. BERT applies bidirectional training of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. Its goal is to generate a language model, so only the encoder mechanism is necessary here.
The model achieved State-of-the-Art result for this chatbot, and I did the model evaluation by using Negative log-likelihood which had minimal averages:

![nll](https://user-images.githubusercontent.com/53462948/165817782-21003bb8-b53e-4b3d-9e3e-b45852beea7e.png)


## Workflow

![Untitled Diagram-Page-1](https://user-images.githubusercontent.com/53462948/165815411-72b2a6ae-ccd2-4c81-92a0-94b1a6d1579c.jpg)


## Data

Dataset can be found in json and excel files, which have been created for this problem. Future improvement include to replace them with any relevant API service.


## Deployment 

I used AWS ECR to create a docker image and push my code into it.  ECR is a fully managed container registry offering high-performance hosting. I also used AWS ECS that is a fully managed container orchestration service to deploy, manage, and scale my containerized application. This allowed me to deploy my application to AWS Elastic Compute Cloud (EC2). The site can be accessed by this [link](http://ec2-54-242-253-116.compute-1.amazonaws.com:8080). The screenshot from the deployed app: 

![aws](https://user-images.githubusercontent.com/53462948/165804296-1b4c1646-3c21-43c9-91d9-e8778e81d264.png)



## Running in the local machine:

1. Clone this repo
2. Download [my trained model](https://drive.google.com/file/d/11zNu1DCuwDSsskYhEgDKynxtFbC0S0DS/view?usp=sharing) and put it in a new "models" folder within the repo directory
3. make install
4. python application.py

## Demo


https://user-images.githubusercontent.com/53462948/165819011-21d83a00-d244-45bb-9246-686f95d0c8db.mov



