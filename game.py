import pygame
from AI import AI
from Stuff import Object
from NN import Brain

pygame.init()


color = (255,255,255)
rect_color = (255,0,0)

pygame.display.set_caption("Show Image")


screen = pygame.display.set_mode((1000, 300))



def checkKeyInputs(keys,dt):


    # Using a switch 


    if keys[pygame.K_w]:
        print("Move the character forwards")
        ai.move(1,dt)
    elif keys[pygame.K_s]:
        print("Move the character backwards")
        ai.move(-1,dt)
    if keys[pygame.K_a]:
        print("Rotate left")
        ai.rotate(-1, dt)
    elif keys[pygame.K_d]:
        print("Rotate rihgt")
        ai.rotate(1, dt)


def checkMouseMotion(event):
      if event.rel[0] > 0:
        print("Mouse moving to the right")
      elif event.rel[1] > 0:
        print("Mouse moving down")


def checkMousePress(event,box):
    if event.button == 1:
        print("Left mouse button pressed")
        mp = pygame.mouse.get_pos()
        print(mp)
        if box.collidepoint(mp):
            ai.feed()
def checkMouseRelease():
    #print("Mouse button has been released")
    pass


TIME_PASSING = pygame.USEREVENT +1


pygame.font.init()
my_font = pygame.font.SysFont('Comic Sans MS', 30)

text_surface = my_font.render('Some Text', False, (0, 0, 0))

font = pygame.font.Font(None, 36)



pygame.time.set_timer(TIME_PASSING, 1000)

box = pygame.Rect(100, 100, 60, 60)
box1 = pygame.Rect(190,190, 60, 60)
ob = Object(box1,(1,1,1), "Food")
wallLeft =  Object(pygame.Rect(0,0, 2, 1000),(-1,-1,1), "Wall")
wallRight =  Object(pygame.Rect(998,0, 2, 1000),(-1,-1,1), "Wall")
wallUpp =  Object(pygame.Rect(0,0, 1000, 2),(-1,-1,1), "Wall")
wallDown =  Object(pygame.Rect(0,298, 1000, 2),(-1,-1,1), "Wall")
objects = [ob, wallRight,wallLeft, wallUpp,wallDown]
ai = AI(objects, lambda x, y, w, h: pygame.Rect(x, y, w, h), Brain())





def menu():
    while True:
        screen.fill(color)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Avslutar")
                return 0
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    return 1
                elif event.key == pygame.K_p:
                    return 2
                elif event.key == pygame.K_f:
                    return 3

    
                


def playing():
    running = True
    ai.load("brain")
    while running:
        screen.fill(color)
        dt = clock.tick(60) / 1000
        key = pygame.key.get_pressed()

        inputData= ai.input(dt)




        # Here is the ai
    
        oldHunger = ai.getHunger()
        move = ai.think(dt)


        ai.movement(move,dt)

        reward = ai.getHunger() - oldHunger
        # print(reward)
        # loss = -log_prob * reward


        checkKeyInputs(key,dt)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Avslutar")
                running = False
            elif event.type == pygame.MOUSEMOTION:
            #checkMouseMotion()
                pass
            elif event.type == pygame.MOUSEBUTTONDOWN:
                checkMousePress(event,box)
            elif event.type == pygame.MOUSEBUTTONUP:
                checkMouseRelease()

            elif event.type == TIME_PASSING:
                #print("1 seconds have passed! (Via Event)")
                pass
        

        
        text = font.render(f"{inputData}", True, (0, 0, 0))
        screen.blit(text, (0,0))

        for obj in objects:
            pygame.draw.rect(screen, rect_color, obj.gameObject)
        
        pygame.draw.rect(screen, rect_color, ai.getBody())
        pygame.draw.line(screen, rect_color, ai.getRayStart(), ai.getRayEnd()) 
        pygame.display.update()


def training():
    running = True
    data = []
    i = 0
    ai.load("brain")

    sumdt = 0
    while running:
        screen.fill(color)
        dt = clock.tick(60) / 1000
        key = pygame.key.get_pressed()
        i+=1

        data.append(ai.liveTrain(dt))
        sumdt += dt

        if i>= 1000:
            ai.trainFromData(data)
            data = []
            i=0
            ai.setPosition((110,110))
            

        checkKeyInputs(key,dt)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Avslutar")
                ai.save()
                running = False
            elif event.type == pygame.MOUSEMOTION:
            #checkMouseMotion()
                pass
            elif event.type == pygame.MOUSEBUTTONDOWN:
                checkMousePress(event, box)
            elif event.type == pygame.MOUSEBUTTONUP:
                checkMouseRelease()

            elif event.type == TIME_PASSING:
                #print("1 seconds have passed! (Via Event)")
                pass
        

        
        text = font.render(f"{ai.getState()}", True, (0, 0, 0))
        screen.blit(text, (0,0))
        for obj in objects:
                    pygame.draw.rect(screen, rect_color, obj.gameObject)
                
        pygame.draw.rect(screen, rect_color, ai.getBody())
        pygame.draw.line(screen, rect_color, ai.getRayStart(), ai.getRayEnd()) 
        pygame.display.update()



def fasttraining():
    running = True
    i= 0
    ai.load("brain")
    while running:
        text = ai.trainUntilWin()
        i+=1
        print(text)
        ai.reset()
        print(ai.getHunger())
        print("Ai moved")
        if i >= 20:
            
        
            i=0

        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                print("Avslutar")
                ai.save()
                running = False
            elif event.type == pygame.QUIT:
                print("Avslutar")
                ai.save()
                running = False


choice = menu()
clock = pygame.time.Clock()

match choice:
    case 0:
        pass
    case 1:
        training()
    case 2:
        playing()
    case 3:
        fasttraining()
        

# Quit Pygame
pygame.quit()