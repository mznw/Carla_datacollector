import math
a = open('steer.txt', 'r').readlines()

delta = [0, 0.1351,  -0.1351,]

for tl in a:
  tl = tl.strip().split()
  if len(tl) != 4: continue
  steer = float(tl[0])
  speed = float(tl[3])
  ta = steer

  for td in delta:
    print(ta + td)
  if speed < 0.01: continue
  ds = math.atan(1.5/(speed / 3.6 * 2)) * 180 / 540
  print(ta + ds)
  print(ta - ds)
  

