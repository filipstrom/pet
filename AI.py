import math

import torch
from NN import Brain
from Stuff import Object

class AI:
    RAYDISTANCE = 200
    TOPPROTATIONSPEED = 180
    TOPSPEED = 100
    HUNGERSPEED= 0.01
    BOREDOMESPEED= 0.01
    STARTPOSITION = (110,110)
    STARTSTATE= {"Hunger": -1.0, "Boredom":-1.0}
    STARTDIRECTION = 100
    def __init__(self, objects: list[Object], makeBody, brain: Brain) -> None:
        self.__direction = self.STARTDIRECTION
        self.__state = self.STARTSTATE.copy()
        self.__position = self.STARTPOSITION
        self.__objects = objects
        self.__makeBody = makeBody
        self.__oldSpeed = 0
        self.__oldRotationSpeed = 0
        self.__speed = 0
        self.__rotationSpeed = 0
        self.__oldState = self.__state.copy()
        self.__brain = brain


    def liveTrain(self, dt):
        action, log_prob = self.__brain.get_action(self.input(dt))

        self.movement(action.tolist(), dt)

        reward = (self.getHunger() - self.__oldState["Hunger"])*10000

        return log_prob, reward
    def setPosition(self,position):
        self.__position = position

    def trainFromData(self, training_data):
        losses = []
        rewards = []

        before = {
        name: param.detach().clone()
        for name, param in self.__brain.named_parameters()
        }
        for log_prob, reward in training_data:
            rewards.append(reward)
            losses.append(-log_prob * reward)

        loss = sum(losses)

        self.__brain.optimizer.zero_grad()
        loss.backward()
        self.__brain.optimizer.step()


        for name, param in self.__brain.named_parameters():
            diff = param.detach() - before[name]
        print(name, diff.abs().max().item())

        print(
        f"Step: {self.__brain.optimizer.state_dict()['state'][0]['step']} | "
        f"Loss: {loss.item():.4f} | "
        f"Avg reward: {sum(rewards)/len(rewards):.4f} | "
        f"Hunger: {self.getHunger():.2f}"
)


        return loss.item()

    def save(self):
        self.__brain.save()

    def load(self,name):
        self.__brain.load_state_dict(torch.load(f"{name}-{self.__brain.INPUTS}x{self.__brain.OUTPUTS}.pth"))
    def trainUntilWin(self):
        losses = []
        rewards = []
        log_probs = []

        i = 0
        print("Training until win"+str(self.getHunger()))
        while self.getHunger() <= 0.8:
            i+=1
            action, log_prob = self.__brain.get_action(self.input(0.02))
            self.movement(action.tolist(),0.02)
            reward = (self.getHunger() - self.__oldState["Hunger"])*5000
            log_probs.append(log_prob)
            rewards.append(reward)
        returns = []

        future_reward = 0

        for reward in reversed(rewards):
            future_reward = reward + 0.99 * future_reward
            returns.append(future_reward)

        losses = []

        for log_prob, future_reward in zip(log_probs, returns):
            losses.append(-log_prob * future_reward)


        loss: torch.Tensor = sum(losses) / len(losses)
        
        self.__brain.optimizer.zero_grad()
        loss.backward()    
        self.__brain.optimizer.step()
        
        
        return f"Step: {self.__brain.optimizer.state_dict()['state'][0]['step']} | "+ f"Loss: {loss.item():.4f} | "+  f"Avg reward: {sum(rewards)/len(rewards):.4f} | "+f"Hunger: {self.getHunger():.2f}" + f"{self.__brain.optimizer.param_groups[0]['lr']}"



    def trainSteps(self, steps):

        losses = []
        rewards = []
        log_probs = []

        i = 0
        for _ in range(steps):
            i+=1
            action, log_prob = self.__brain.get_action(self.input(0.02))
            self.movement(action.tolist(),0.02)
            reward = (self.getHunger() - self.__oldState["Hunger"])*5000
            log_probs.append(log_prob)
            rewards.append(reward)
        returns = []

        future_reward = 0

        for reward in reversed(rewards):
            future_reward = reward + 0.99 * future_reward
            returns.append(future_reward)

        losses = []

        for log_prob, future_reward in zip(log_probs, returns):
            losses.append(-log_prob * future_reward)


        loss: torch.Tensor = sum(losses) / len(losses)
        
        self.__brain.optimizer.zero_grad()
        loss.backward()    
        self.__brain.optimizer.step()
        
        
        return f"Step: {self.__brain.optimizer.state_dict()['state'][0]['step']} | "+ f"Loss: {loss.item():.4f} | "+  f"Avg reward: {sum(rewards)/len(rewards):.4f} | "+f"Hunger: {self.getHunger():.2f}" + f"{self.__brain.optimizer.param_groups[0]['lr']}"

    def think(self, dt):
        inputData = self.input(dt)
        output = self.__brain.forward(inputData).tolist()
        return output

    def printState(self):
        print(self.__state)

    def getState(self):
        stats = ""
        for key, value in self.__state.items():
            stats += f"{key}:{value:0.2f}\n"

        stats += f"Rotation: {self.__direction}"
        return stats

    def feed(self):
        if self.__increaseState("Hunger", 0.2):
            print("You fedd it a cookie")


    def __increaseState(self, state, value):
        startingPoint = self.__state[state]
        value = abs(value)
        newValue = startingPoint + value
        if newValue > 1:
            #print(f"oh, the {state} it can't increase any higher")
            return False
        else:
            self.__oldState[state] = startingPoint
            self.__state[state]=newValue
            return True

    def __decrease(self, state, value):
        startingPoint = self.__state[state]
        value = abs(value)
        newValue = startingPoint - value
        if newValue < -1:
            # print(f"oh, the {state} it can't decrease any lower")
            return False
        else:
            self.__oldState[state] = startingPoint
            self.__state[state]=newValue
            return True

    def getPosition(self):
        return self.__position

    def getBody(self):
        # print(self.__position[0], self.__position[1], 60, 60)
        return self.__makeBody(self.__position[0], self.__position[1], 60, 60)
    
    def getDirection(self):
        return self.__direction
    def getHunger(self):
        return self.__state["Hunger"]

    def move(self, movement,dt):
        self.__position = (self.__position[0] + movement*self.TOPSPEED *dt*math.cos(math.radians(self.__direction)), self.__position[1] + movement*dt*self.TOPSPEED*math.sin(math.radians(self.__direction))) 
        for obj in self.__objects:
            if obj.gameObject.colliderect(self.getBody()):
                if obj.name == "Food":
                    self.__increaseState("Hunger",0.1*dt)
                # print("Collision detected!")
                self.__position = (self.__position[0] - movement*dt*self.TOPSPEED *math.cos(math.radians(self.__direction)), self.__position[1] - movement*dt*self.TOPSPEED*math.sin(math.radians(self.__direction)))
                return
            else:
                self.__oldSpeed = self.__speed
                self.__speed = movement
                

    def rotate(self, speed, dt):
        self.__direction += speed*self.TOPPROTATIONSPEED * dt
       
        self.__oldRotationSpeed = self.__rotationSpeed
        self.__rotationSpeed = speed
        if self.__direction >= 360:
            self.__direction -= 360
        elif self.__direction < 0:
            self.__direction += 360

    def getRayEnd(self):
        end_x = self.getRayStart()[0] + self.RAYDISTANCE * math.cos(math.radians(self.__direction))
        end_y = self.getRayStart()[1] + self.RAYDISTANCE * math.sin(math.radians(self.__direction))
        return (end_x, end_y)

    def getRayStart(self):
        return (self.getPosition()[0]+ 30 ,self.getPosition()[1]+30)

    def raycast(self):
        start = self.getRayStart()
        end = self.getRayEnd()

        hits = []
        for obj in self.__objects:
            hit = obj.gameObject.clipline(start, end)
            if hit:
                #print("Ray hit object!")
                hits.append((math.dist(start, hit[0]),obj))
            
        if len(hits) == 0:
            return (-1, (-1,-1,-1))
        #print(type(hits[0][0]))
        hits.sort(key=lambda x: x[0])

        closest = hits[0]
        return ((closest[0]/self.RAYDISTANCE)*-2+1, closest[1].apperance)

    def reset(self):
        self.setPosition(self.STARTPOSITION)

        self.__state = self.STARTSTATE.copy()
        self.__oldState = self.STARTSTATE.copy()

        self.__speed = 0
        self.__oldSpeed = 0
        self.__rotationSpeed = 0
        self.__oldRotationSpeed = 0
        self.__direction = self.STARTDIRECTION
        
    def input(self,dt):
        inp = []
        for key, value in self.__state.items():
            inp.append(value)
            inp.append(self.__oldState[key])
            inp.append((value-self.__oldState[key])/dt)
        a, b = self.raycast()
        inp.append(a)
        inp.extend(b)
        inp.append(self.__direction/360*2-1)
        inp.append(self.__oldSpeed)
        inp.append(self.__oldRotationSpeed)
        inp.append(self.__speed)
        inp.append(self.__rotationSpeed)
        # inp.append((self.__rotationSpeed-self.__oldRotationSpeed)/dt)
        # inp.append((self.__speed-self.__oldSpeed)/dt)

       

        return inp

    def movement(self, output,dt):
        self.__decrease("Hunger", self.HUNGERSPEED*dt)
        self.__decrease("Boredom", self.BOREDOMESPEED*dt)
        self.move(output[0],dt)
        self.rotate(output[1],dt)

        
        