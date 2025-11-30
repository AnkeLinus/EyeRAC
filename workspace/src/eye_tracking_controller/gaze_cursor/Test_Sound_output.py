import pygame 
import time
#playsound("~/usr/share/sounds/sound-icons/prompt.wav")
       
        
def main(args) -> None: 
    # For Measurement
    pygame.init()
    pygame.mixer.init()
    plays = pygame.mixer.Sound("/usr/share/sounds/gnome/default/alerts/drip.ogg")
    plays.play()
    time.sleep(1)
    #while pygame.mixer.music.get_busy():
    #    pygame.event.pump()


if __name__ == "__main__":
    

    main(args = None)
