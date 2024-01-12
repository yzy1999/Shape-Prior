import numpy as np
from scipy.special import sph_harm

l1 = 0
m1 = 0

l2 = 1
m2 = -1

l3 = 1
m3 = 0

l4 = 1
m4 = 1

l5 = 2
m5 = -2

l6 = 2
m6 = -1

l7 = 2
m7 = 0

l8 = 2
m8 = 1

l9 = 2
m9 = 2

#in scipy, theta is azimuthal angle and phi is polar angle
theta = np.linspace(0, 2 * np.pi, 61)
phi = np.linspace(0, np.pi, 31)
theta_3d, phi_3d = np.meshgrid(theta, phi)

xyz_3d = np.array([np.sin(phi_3d) * np.sin(theta_3d),
                   np.sin(phi_3d) * np.cos(theta_3d),
                   np.cos(phi_3d)])

Ylm1 = sph_harm(abs(m1), l1, theta_3d, phi_3d)
if m1 < 0:
    Ylm1 = np.sqrt(2) * (-1)**m1 * Ylm1.imag
elif m1 > 0:
    Ylm1 = np.sqrt(2) * (-1)**m1 * Ylm1.real
r1 = np.abs(Ylm1.real) * xyz_3d
print(r1.shape)

Ylm2 = sph_harm(abs(m2), l2, theta_3d, phi_3d)
if m2 < 0:
    Ylm2 = np.sqrt(2) * (-1)**m2 * Ylm2.imag
elif m2 > 0:
    Ylm2 = np.sqrt(2) * (-1)**m2 * Ylm2.real
r2 = np.abs(Ylm2.real) * xyz_3d
print(r2.shape)

Ylm3 = sph_harm(abs(m3), l3, theta_3d, phi_3d)
if m3 < 0:
    Ylm3 = np.sqrt(2) * (-1)**m3 * Ylm3.imag
elif m3 > 0:
    Ylm3 = np.sqrt(2) * (-1)**m3 * Ylm3.real
r3 = np.abs(Ylm3.real) * xyz_3d
print(r3.shape)

Ylm4 = sph_harm(abs(m4), l4, theta_3d, phi_3d)
if m4 < 0:
    Ylm4 = np.sqrt(2) * (-1)**m4 * Ylm4.imag
elif m4 > 0:
    Ylm4 = np.sqrt(2) * (-1)**m4 * Ylm4.real
r4 = np.abs(Ylm4.real) * xyz_3d
print(r4.shape)

Ylm5 = sph_harm(abs(m5), l5, theta_3d, phi_3d)
if m5 < 0:
    Ylm5 = np.sqrt(2) * (-1)**m5 * Ylm5.imag
elif m5 > 0:
    Ylm5 = np.sqrt(2) * (-1)**m5 * Ylm5.real
r5 = np.abs(Ylm5.real) * xyz_3d
print(r5.shape)

Ylm6 = sph_harm(abs(m6), l6, theta_3d, phi_3d)
if m6 < 0:
    Ylm6 = np.sqrt(2) * (-1)**m6 * Ylm6.imag
elif m6 > 0:
    Ylm6 = np.sqrt(2) * (-1)**m6 * Ylm6.real
r6 = np.abs(Ylm6.real) * xyz_3d
print(r6.shape)

Ylm7 = sph_harm(abs(m7), l7, theta_3d, phi_3d)
if m7 < 0:
    Ylm7 = np.sqrt(2) * (-1)**m7 * Ylm7.imag
elif m7 > 0:
    Ylm7 = np.sqrt(2) * (-1)**m7 * Ylm7.real
r7 = np.abs(Ylm7.real) * xyz_3d
print(r7.shape)

Ylm8 = sph_harm(abs(m8), l8, theta_3d, phi_3d)
if m8 < 0:
    Ylm8 = np.sqrt(2) * (-1)**m8 * Ylm8.imag
elif m8 > 0:
    Ylm8 = np.sqrt(2) * (-1)**m8 * Ylm8.real
r8 = np.abs(Ylm8.real) * xyz_3d
print(r8.shape)

Ylm9 = sph_harm(abs(m9), l9, theta_3d, phi_3d)
if m9 < 0:
    Ylm9 = np.sqrt(2) * (-1)**m9 * Ylm9.imag
elif m9 > 0:
    Ylm9 = np.sqrt(2) * (-1)**m9 * Ylm9.real
r9 = np.abs(Ylm9.real) * xyz_3d
print(r9.shape)

n = 10000 # n input points
x = np.random.uniform(0, 1, [n,9]) #lower bound, higher bound, size
print(x.shape)
sigma = 0.1
y = np.zeros([n, 3, 31, 61])

for i in range(n):
    y[i] = x[i,0]*r1 + x[i,1]*r2 + x[i,2]*r3 + x[i,3]*r4 + x[i,4]*r5 + x[i,5]*r6 + x[i,6]*r7 + x[i,7]*r8 + x[i,8]*r9 + np.random.normal(0, sigma)

print(y.shape)
print(y)
np.save('D:/shape_prior_data/x_advanced_sigma01.npy',x)
np.save('D:/shape_prior_data/y_advanced_sigma01.npy',y)