
import hparams
import sys
import torch
import datetime
sys.path.append("./")

class Metropolis():
    # move ramdonly act on one of the unit in configuration
    # fn is the function to be integraled
    def __init__(self,configuration,fn,move,ramdon_deep,T,kB):
        assert torch.is_tensor(configuration)
        self.params = hparams.create_hparams()
        self.configuration = configuration
        self.move = move
        self.deep = ramdon_deep
        self.fn = fn
        self.points = []
        self.T = T
        self.kB = kB
        _points = []
        for _ in range(0,self.params['points_number']):
            for d in range(0,ramdon_deep):
                _points += torch.randint(self.configuration.shape[d], (1,))
            self.points.append(_points)
            _points = []
        
        self.points = torch.tensor(self.points)
        
        # print(self.points)
        # print(self.points.shape)

        if self.params['gpu']:
            if torch.cuda.is_available():
                self.configuration = self.configuration.to('cuda')
                # self.points = self.points.to('cuda')
                print(f"Device tensor is stored on: {self.configuration.device}")
            else:
                print(f"no gpu found, cpu is used instead")
        
        start = datetime.datetime.now()
        self.E  = self.fn(self.configuration,globally=True)
        end = datetime.datetime.now()
        print(self.E,end-start)

    def forward(self):
        assert self.deep == 2
        for point in self.points:
            forward_flag = False
            Eb = self.fn(self.configuration,globally=False,unit=point)
            nextStep = self.configuration.clone()
            nextStep[point[0],point[1]] = self.move(self.configuration[point[0],point[1]])
            Ea = self.fn(self.configuration,globally=False,unit=point)
            if Eb > Ea:
                forward_flag = True
            elif torch.rand(1) <= torch.exp((Eb - Ea)/(self.kB*self.T)):
                forward_flag = True
            if forward_flag:
                self.configuration = nextStep
                self.E = self.E - Ea
            print(Ea,Eb)
        pass
    
    def solve(self,epoch):
        while epoch > 0:
            self.forward()
            epoch -= 1

    def summary(self):
        print(self.configuration)
        print("E: ",self.E)



        



