import numpy as np
import pylab as plt
import math
import random

#constants (cgs)
G = 6.67e-8
pc = 1.496e13
Msun = 1.99e33
km = 10**6
year = 3.15e7



#simulation definitions
dt = 0.01*year
tFinal = 5*year
Nstars = 100
#softening length
epsilon = 0.1*pc


class Star():

	def __init__(self):
		Radius = getRndFunc(200,0.1,20.)
		Theta = random.random()*2*math.pi
		Phi = random.random()*math.pi
		self.r = np.array([Radius*math.cos(Theta)*math.sin(Phi)*pc, Radius*math.sin(Theta)*math.sin(Phi)*pc, random.random()*2*pc])
		vel = random.random()/Radius
		self.v = np.array([vel*km*math.sin(Theta),vel*km*math.cos(Theta),0])

		self.mass = random.random()*Msun
		
	def update_velocity(self, dt, totalForce):
		a = totalForce / self.mass
		self.v += a * dt
		
	def update_position(self, dt):
		self.r += self.v * dt
		
	def draw(self):
		plt.plot(self.r[0]/pc, self.r[1]/pc, 'k*')
		
		

		

def weighted_choice(weights):
    totals = []
    running_total = 0

    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    for i, total in enumerate(totals):
        if rnd < total:
            return i



def getRndFunc(nbins,min,max):
	bins = (np.arange(min,nbins)* max/nbins)
	prob = 10/np.sqrt(bins**2 + 2**2)
	choice = weighted_choice(prob)
	
	variation = (max - min) / nbins * (random.random()-0.5)
	
	return bins[choice] + variation


starsx = np.array([])
starsy = np.array([])

# 
# for ii in range(1000):
# 	starsRadius = getRndFunc(200,0.1,10.)
# 	starsTheta = random.random()*2*math.pi
# 	starsx = np.append(starsx, starsRadius*math.cos(starsTheta))
# 	starsy = np.append(starsy, starsRadius*math.sin(starsTheta))
# 	
	
	
#initialize stars
Stars = [Star() for ii in range(Nstars)]
	

			

def update_all_velocities(Nstars, dt):
	for thisStar in range(Nstars):
		totalForce = 0
		for otherStar in range(Nstars):
			if thisStar != otherStar:
				r = Stars[otherStar].r - Stars[thisStar].r
				rmagSqr = np.sum(r**2)
				rmag = np.sqrt(rmagSqr)
				rUnit = r / rmag
				
				Force_thisStar = (G*Stars[thisStar].mass*Stars[otherStar].mass) * rUnit / (rmagSqr + (epsilon**2))
				totalForce += Force_thisStar
		
		Stars[thisStar].update_velocity(dt,totalForce)
		Stars[thisStar].update_position(dt)


			
			

	
#Main time loop
#create the plotting window
for t in range(int(tFinal/dt)):
	update_all_velocities(Nstars,dt)
	for star in Stars:
		star.draw()
	
	print t
	plt.xlim([-20,20])
	plt.ylim([-20,20])
	plt.savefig("galaxy/"+str(t)+'.png', bbox_inches='tight')

# 	plt.draw()
# 
# 	plt.pause(0.1)
 	plt.clf()
	
	
	
	
	
# for ii in range(len(starsx)):
# 	plt.plot(starsx[ii],starsy[ii],'k*')


# plt.xlim([-10,10])
# plt.ylim([-10,10])
# 
# plt.show()