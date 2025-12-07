import sys

def check(n):
  print(n)
  print('k')
  if n.isnumeric():
    i = int(n)
    if i >= 0 and i <= 20000:
      return i
    else:
      sys.stdout.write("0")
      sys.exit()
  else:
    sys.stdout.write("0")
    sys.exit()
  
n = sys.stdin.readline()
if n.isnumeric():
  n = int(n)
  if n < 0 and n > 2000:
    sys.stdout.write("0")
    sys.exit()

n = int(n)
s = set()
for _ in range(0, n):
  i = check(sys.stdin.readline())
  s.add(i)

s = sorted(s)
max = s.pop() - s[0]
sys.stdout.write(str(max))
