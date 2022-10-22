import subprocess
from atexit import register
import socket
import os

rootpath = os.path.abspath(os.path.dirname(__file__)) + "/"

def clean(prog):
    try:
        prog.close()
    except:
        pass

# start a server for the Mario-AI Framework
class AStarAgentServer():
    '''
    Comunicate with a java program located in  pcg_gym/envs/Mario-AI-Framework-master.jar
    The program contains a mario game with an A* agent to test the generated segments.
    '''

    def __init__(self, visuals):
        # self.id = id
        # self.test_id = 0
        self.visuals = '1' if visuals else '0'

    def start(self):
        # register(clean, self)
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # find an unused port
        sock = socket.socket()
        sock.bind(('', 0))
        self.port = int(sock.getsockname()[1])
        sock.close()

        print("Port Number: ", self.port)

        # os.system("java -jar "+rootpath + "/Mario-AI-Framework-master.jar " + rootpath +
        #           "/test_pool/" + str(self.id) + " " + str(self.port)+" "+self.visuals+" &")

        # 1st argument = map file path
        # 2nd argument = port number
        # 3rd arugment = visuals bool
        subprocess.call(['java', '-jar', rootpath + "Mario-AI-Framework.jar", 
                        rootpath + "mario_current_map.txt", str(self.port), self.visuals])

        # while True:
        #     try:
        #         self.client.connect(('localhost', self.port))
        #         break
        #     except:
        #         pass

    # def close(self):
    #     msg = 'close'
    #     self.client.send(msg.encode('utf-8'))
    #     self.client.close()
    #     # self.test_id = 0

    # def start_test(self, lv):
    #     '''
    #     Test the first segment. The initial position of Mario is set as default
    #     '''
    #     return self.__test(lv, "start")

    # def continue_test(self, lv):
    #     '''
    #     Used when the previous segment is playable.
    #     The initial postion of Mario is set according to previous test. 
    #     '''
    #     return self.__test(lv, "continue\n")

    # def retest(self, lv):
    #     '''
    #     Used when the previous segment is unplayable.
    #     The initial postion of Mario keeps the same as the previous test. 
    #     '''
    #     return self.__test(lv, "retest\n")

    # def __test(self, lv, msg):
    #     # name = str(self.id)+"_"+str(self.test_id)
    #     # saveLevelAsText(lv, rootpath+"/test_pool/"+name)
    #     saveLevelAsText(lv, rootpath + "mario_current_map")
    #     res = self.client.send(msg.encode('utf-8'))
    #     data = self.client.recv(1024)
    #     rate = float(data.decode())
    #     # os.remove(rootpath+"/test_pool/"+name+".txt")
    #     # self.test_id += 1
    #     return rate

server = AStarAgentServer(0)
server.start()