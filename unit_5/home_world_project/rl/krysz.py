import numpy as np
import time
a = 'Krysztof'
b = 'Krysztof'

imiona = ['Marino', 'Krysztof', 'Antek', 'Justyna','Marek', 'Karolina', 'Emma']


for imie in imiona:
	print(imie)



mocna = 100

jest_marino_kobieta = False
jest_justyna_kobieta = True


while jest_justyna_kobieta:
	print('Justina jest kobieta')




class Ludze:

	def __init__(self, imie):
		self.imie = imie
		self.zycie = 100

	def attack(self, other):
		self.moc = np.random.randint(0,10)
		other.zycie = other.zycie - self.moc
		print(f'{self.imie} atak {other.imie} z uszkodzeniem {self.moc}')
		if other.zycie <= 0:
			print(f'{other.imie} został zabity do {self.imie}')


k = Ludze('Krysztof')
m = Ludze('Marino')

while k.zycie > 0 and m.zycie >0:
	k.attack(m)
	m.attack(k)
	print(f'{m.imie} ma {m.zycie}')
	print(f'{k.imie} ma {k.zycie}')