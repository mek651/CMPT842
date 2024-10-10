# CMPT842 Project

## Federated Learning with Flower: A Dockerized Approach for Privacy-Preserving Image Classification
Federated Learning (FL) is a distributed machine learning framework enabling multiple clients to train a model collaboratively without sharing their local data. This approach mitigates privacy concerns that arise in traditional centralized learning models. In this paper, we present the application of the Flower framework for an Image Classification task. Flower, a flexible and scalable federated learning platform, facilitates experimentation with heterogeneous devices and machine learning frameworks. Our methodology leverages the CIFAR-10 dataset to train a Convolutional Neural Network (CNN) model in a federated setting, using Docker for simulation and deployment. The experimental setup involves partitioning data across clients, configuring server and client applications, and evaluating model performance. The results demonstrate the effectiveness of federated learning in maintaining model accuracy while preserving data privacy. This study highlights the importance of FL in scenarios where data centralization is not feasible due to privacy regulations or practical constraints. This hands-on approach showcases the critical role of Dockerization in simplifying deployment, scaling, and managing services while addressing modern computational challenges like data privacy and decentralized machine learning.

## Run Project
To run the project you need to perform the following steps:
1- Clone this repository to your local machine
2- After cloning, go to path "..../complete"
3- Inside the "complete" directory open a terminal and run the following commands in order to build the docker
  * export PROJECT_DIR=quickstart-compose
  * docker compose -f compose.yml up --build -d

Now all of the Containers starts in a detached form

4-Open another terminal and run the following command to run the simulation:
  * flwr run quickstart-compose docker-compose

5- To monitor the SuperExec logs and see the summary on the server side run the following command:
  * docker compose logs superexec -f

6- To monitor the logs for all clients and server run the following command:
  * docker compose logs -f
