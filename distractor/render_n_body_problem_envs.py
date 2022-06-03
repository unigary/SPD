from n_body_problem import Planets, Electrons, IdealGas


# env1 = Planets(num_bodies=10, num_dimensions=2, dt=0.01, contained_in_a_box=True)
# env1.animate()  # only animates if num_dimensions == 2
#
# env2 = Electrons(num_bodies=10, num_dimensions=2, dt=0.01, contained_in_a_box=True)
# env2.animate()  # only animates if num_dimensions == 2

for i in range(100):
    env3 = IdealGas(num_bodies=10, num_dimensions=2, dt=0.01, contained_in_a_box=True)
    file_name = '/home/kim/drq/resource_folder/idealgas{}.mp4'.format(i)
    env3.animate(file_name=file_name, pixel_length=64)