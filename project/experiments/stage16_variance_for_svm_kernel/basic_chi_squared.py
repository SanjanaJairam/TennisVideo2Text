from __future__ import division

histogram_x = [1,2]
histogram_y = [3,4]

def chisquared(x,y):
	distance = 0
	for i in range(len(x)):
		distance += round(((x[i]-y[i])**2)/((x[i]+y[i])*2),4)
		print(distance)
	return distance
	
print(chisquared(histogram_x,histogram_y))