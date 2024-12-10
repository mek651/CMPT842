# CMPT842 Project

## Federated Learning with Flower: A Dockerized Approach for Privacy-Preserving Image Classification
Federated Learning (FL) is a distributed machine learning framework enabling multiple clients to train a model collaboratively without sharing their local data. This approach mitigates privacy concerns that arise in traditional centralized learning models. In this paper, we present the application of the Flower framework for an Image Classification task. Flower, a flexible and scalable federated learning platform, facilitates experimentation with heterogeneous devices and machine learning frameworks. Our methodology leverages the CIFAR-10 dataset to train a Convolutional Neural Network (CNN) model in a federated setting, using Docker for simulation and deployment. The experimental setup involves partitioning data across clients, configuring server and client applications, and evaluating model performance. The results demonstrate the effectiveness of federated learning in maintaining model accuracy while preserving data privacy. This study highlights the importance of FL in scenarios where data centralization is not feasible due to privacy regulations or practical constraints. This hands-on approach showcases the critical role of Dockerization in simplifying deployment, scaling, and managing services while addressing modern computational challenges like data privacy and decentralized machine learning.

## Deploy Flower on a Single Machine with Docker Compose
To run the project you need to perform the following steps:

1- Clone this repository to your local machine

2- After cloning, go to path "..../Single_Machine_Flwr"

3- Inside the "Single_Machine_Flwr" directory open a terminal and run the following commands in order to build the docker
  * export PROJECT_DIR=quickstart-compose
  * docker compose -f compose.yml up --build -d

Use the following command to see all the generated containers
  * docker ps


Now all of the Containers starts in a detached form

4-Open another terminal and run the following command to run the simulation:
  * flwr run quickstart-compose docker-compose

5- To monitor the SuperExec logs and see the summary on the server side run the following command:
  * docker compose logs superexec -f

6- To monitor the logs for all clients and server run the following command:
  * docker compose logs -f



## Deploy Flower on Multiple Machine with Docker Compose
To run the project on 2 machines (one as the server and another as the clients) you need to perform the following steps:

### Step 1: Set Up
1- On the machine considered as the client, clone this repository to your local machine

2- After cloning, go to path "..../Multi_Machine_flwr"

3- Then, by running the following command go to the "distributed" directory:
   
   * cd flower/src/docker/distributed

4- Need to get the IP address from the remote (server) machine and save it for later. For example: 192.168.2.33

5- Set the environment variable SUPERLINK_IP with the IP address from the remote machine using the following command:

   * export SUPERLINK_IP=192.168.2.33

6- Generate the self-signed certificates:
   
   * docker compose -f certs.yml -f ../complete/certs.yml run --rm --build gen-certs



### Step 2: Copy the Server Compose Files

7- Copy the server directory, the certificates, and the pyproject.toml file of the Flower project to the remote (server) machine using the following command:
   
   * scp -r ./server \
       ./superlink-certificates \
       ../../../examples/quickstart-sklearn-tabular/pyproject.toml remote:~/distributed


### Step 3: Start the Flower Server Components (on the server machine)

8- On the server machine run the following command to start the SuperLink and ServerApp services:
   
   * cd ./server
   
   * export PROJECT_DIR=../
   
   * docker compose -f compose.yml up --build -d

9- you may need to create the state directory first and change its ownership to ensure proper access and permissions. 

After exporting the PROJECT_DIR (and before docker compose), run the following commands:
   
   * mkdir ./state
   * sudo chown -R 49999:49999 ./state


### Step 4: Start the Flower Client Components (back to the client machine)

10- On the  local machine, run the following command to start the client components:
   
  * export PROJECT_DIR=../../../../examples/quickstart-sklearn-tabular
  * docker compose -f client/compose.yml up --build -d

Use the following command to see all the generated containers
  * docker compose -f client/compose.yml ps

Use the following command to see all the logs in live
  * docker compose -f client/compose.yml logs -f


### Step 5: Run Your Flower Project
11- Specify the remote SuperLink IP addresses and the path to the root certificate in the [tool.flwr.federations.remote-deployment] table in the pyproject.toml file. 
name the remote federation remote-deployment.

So open pyproject.toml file and at the end of the file add the following 3 lines:

    [tool.flwr.federations.remote-deployment]
    address = "192.168.2.33:9093"
    root-certificates = "../../src/docker/distributed/superlink-certificates/ca.crt"


12- Run the project and follow the ServerApp logs using the following command:
   
   * flwr run ../../../examples/quickstart-sklearn-tabular remote-deployment --stream


### Step 6: Clean Up
13- Shut down the Flower client components:
   
   * docker compose -f client/compose.yml down
   * docker compose -f client/compose.yml stop

14- Shut down the Flower server components and delete the SuperLink state (on the server machine)
   
   * cd ./server
   * docker compose -f compose.yml down -v

