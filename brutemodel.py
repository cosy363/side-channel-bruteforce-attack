# import character_model as ch
# import num_keys as n
import numpy as np
from pathlib import Path
import requests
import random
import os
from threading import Thread
from time import time, sleep

# 초기설정
password = ['d2$s']
passworddir = Path.cwd() / 'testfile'   # wav data 경로
fp_list = list(passworddir.rglob("*.wav"))   # wav list

#Brute Force Model
char = "abcdefghijklmnopqrstuvwxyz0123456789"
spec_char = "'!#$%&'("
totalchar = "abcdefghijklmnopqrstuvwxyz0123456789'!#$%&'("

ans_list = [3, 1, 0, 7, 5, 9, 8, 4, 6, 10, 11, 2, 13, 14, 12, 15]
number = 4

def main():
    global Realpassword
    global trylist
    global fin
    trylist = []
    
    reinforced = False
    fin = False
    while True:
        if fin is True:
            break
        valid = False

        while not valid:
            specialchoice = ans_list[0]
            choicenum1 = number - specialchoice
            choicenum2 = specialchoice - 1

            if reinforced is False:
                randpass = random.choices(totalchar,k=number)
            else:
                if specialchoice == 0:
                    randpass = random.choices(char, k=number)
                else:
                    if choicenum1 == 0:
                        randpass2 = random.choices(char, k=choicenum2)
                        specialchoice = random.choices(spec_char, k=1)
                        randpass = specialchoice + randpass2 
                    else:
                        randpass1 = random.choices(char, k=choicenum1)
                        randpass2 = random.choices(char, k=choicenum2)
                        specialchoice = random.choices(spec_char, k=1)
                        randpass = randpass2 + specialchoice + randpass1
             
            passwrd = "".join(randpass)
            if passwrd in trylist:
                pass
            else:
                valid = True
                print(passwrd + " == " + password[0] + "?")

        if password[0] == passwrd:
            Realpassword = passwrd
            fin = True
            print("심봤다!!")
        else:
            trylist.append(passwrd)
            with open("tries.txt", "a") as f:
                f.write(f"{passwrd}\n")


# Threading and timing and printing
start = time()
for x in range(20):
    Thread(target=main).start()

while True:
    if fin is True:
        end = time()
        break
    else:
        sleep(0.01)

timer = end - start

sleep(10)
print("Attack Time: "+str(timer))            
print("Password: " + str(Realpassword))    