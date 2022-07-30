"""
QNCP Crystal Optimization.
This is used to find a near-optiminal poles for on a great circle for the Lithium Niobate Polarization Controller 
by EOSPace. See: https://www.eospace.com/polarization-controller.

There are several steps involved.
1) The characterization of the crystal. Voltages are swept through the crystal and the output polarization is measured. The function `characterize_crystal` provides this functionality.
2) Fitting a model to the crystal. Machine learning is used to fit a model to interpolate the polarization between the voltages used to characterize the great circle. This step can be used to improve the fit, but may optionally be omitted. The function `fit_crystal_model` provides this functionality.
3) Finding near-optiminal poles on a great circle. The measured polarizations coupled with the interpolated polarizations are used to find four poles (north, south, east, west) on the great circle. The poles and the voltages required to produce those poles are reported. The function `find_poles` provides this functionality.
"""
import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

def characterize_crystal(
        set_v1, 
        set_v2, 
        measure_polarization, 
        v_1_range=(0, 8), 
        v_2_range=(0, 30), 
        v_1_n_samples=30, 
        v_2_n_samples=30,
        polarization_n_samples=2
):
    """
    Perform the characterization of the crystal.
    `set_v1`, `set_v2`, and `measure_polarizations` are functions that abstract away
    the specific details of the power supply and polarization devices used.
    `set_v1` should set V1 to the provided voltage (in volts).
    `set_v2` should set V2 to the provided voltage (in volts).
    `measure_polarization` should return the last three components of a normalized stokes vector.

    The user is responsible for ensuring that enough time is passed between setting the voltages
    and measuring the polarizations, through the use of a sleep() call in the measure_polarization
    function.

    `v_1_range` and `v_2_range` specify the range of the voltages explored for v_1 and v_2, respectively. 
    They are each a 2-tuple with the first entry indicating the minimum voltage, in volts, and the
    second entry indicating the maximum voltage, in volts.

    `v_1_n_samples` and `v_2_n_samples` indicate how many samples to take. The samples are spaced
    linearly using np.linspace. The total number of voltage samples used will
    be `v_1_n_samples` * `v_2_n_samples`.

    `polarization_n_samples` indicate how many times to sample the polarizations for each pair of voltages
    (v_1, v_2). This can help eliminate the effect of noise.

    Returns: samples
    `samples`: A NxD numpy array with the samples. The first two columns are `V_1` and `V_2`. 
    The next three columns are the last three components of the stokes vector. The next columns are 
    any additional measurements that the polarimeter might have returned, such as the power. So long
    as the first 5 columns are (v_1, v_2, s_1, s_2, s_3) in that order, the remaining columns can
    be used to store additional data. 
    """
    samples = []
    with tqdm.tqdm(total=polarization_n_samples*v_1_n_samples*v_2_n_samples) as pbar:
        for v_1 in np.linspace(v_1_range[0], v_1_range[1], v_1_n_samples):
            set_v1(v_1)
            for v_2 in np.linspace(v_2_range[0], v_2_range[1], v_2_n_samples):
                set_v2(v_2)
                for i in range(polarization_n_samples):
                    s = measure_polarization()
                    samples.append((v_1, v_2) + s)
                    pbar.update(1)

    samples = np.stack(samples, axis=0)
    return samples

def fit_crystal_model(data, size=16, epochs=30, lr=1e-2):
    """
    Fit a model to the crystal. 
    
    `data` is an NxD numpy array with the measurements performed on the
    crystal. The first 5 columns of data should be (v_1, v_2, s_1, s_2, s_3).
    `v_1` and `v_2` are the voltages applied to the crystal. `s_1`, `s_2`, and
    `s_3` are the last three components of the noramlized 4-D stokes vector.
    All columns past the first 5 are ignored and may be used to store
    additional data given to the polarimeter. The `data` variable may be
    obtained by using the `characterize_crystal` function provided by the
    package.

    `size` is the size of the hidden dimensions used by the model.
    `epochs` is the number of epochs used to train the model.
    `lr` is the learning rate of the SGD optimizer used to train the model.

    Returns: model
    `model`: A fitted PyTorch model. The (forward) method may be used order
    with a Nx2 tensor in order to obtain predictions for the polarization. The
    columns of the Nx2 tensor are `v_1` and `v_2`, the voltages applied to the
    crystal. The predictions are an Nx3 tensor, with the columns being `s_1`,
    `s_2`, and `s_3`. 
    """

    def lr_schedule(epoch):
        if epoch > 20 and epoch <= 40:
            return 0.1
        elif epoch > 40 and epoch <= 80:
            return 0.01
        elif epoch > 80 and epoch <= 160:
            return 0.001
        elif epoch > 160:
            return 0.0001
        else:
            return 1

    try:
        import torch
        import torch.nn as nn
        from .models import Crystal_Model
        import torch.utils.data as udata
    except ImportError:
        raise ImportError("PyTorch must be installed for crystal-fitting.")

    u = torch.from_numpy(data[:, :2]).to(torch.float)
    s = torch.from_numpy(data[:, 2:5]).to(torch.float)
    perm = np.random.permutation(u.shape[0])

    train_dataset = udata.TensorDataset(u, s)
    train_dataloader = udata.DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = Crystal_Model(size)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    for i in range(epochs):
        total_loss = 0
        for u, s in tqdm.tqdm(train_dataloader):
            s_pred = model(u)
            minibatch_loss = loss(s, s_pred)
            total_loss += minibatch_loss
            opt.zero_grad()
            minibatch_loss.backward()
            opt.step()
        print(f'Epoch {i + 1}: Loss = {total_loss.detach().item()}')
        lr_scheduler.step()

    return model

def find_poles(
        data, 
        model=None, 
        v_1_range=(0, 8), 
        v_2_range=(0, 30), 
        v_1_n_samples=1000, 
        v_2_n_samples=1000,
        ):
    """
    Find a four near-optimal poles `p_1`, `p_2`, `p_3`, and `p_4` producible by the crystal.
    The optimal poles lie on a great circle on the Poincaré sphere and each have an angle of 
    π/4 between them. The near-optimal poles will lie close to this near-optimal set of poles.
    The runtime of this algorithm is O(N^2), where N is the number of data-points given. It
    can comfortably handle ~1800 datapoints in reasonable time.

    `data` is an NxD numpy array with the measurements performed on the
    crystal. The first 5 columns of data should be (v_1, v_2, s_1, s_2, s_3).
    `v_1` and `v_2` are the voltages applied to the crystal. `s_1`, `s_2`, and
    `s_3` are the last three components of the noramlized 4-D stokes vector.
    All columns past the first 5 are ignored and may be used to store
    additional data given to the polarimeter. The `data` variable may be
    obtained by using the `characterize_crystal` function provided by the
    package.

    `model` is a PyTorch model that performs interpolation for the polarization
    between the measure set of points. The `forward` method of the model is called
    with a NxD tensor obtain an Nx3 tensor with the interpolated (s_1, s_2, s_3). It
    an optional parameter and if it is not given, no interpolation will be used in order
    to try to obtain a better set of poles.


    `v_1_range` and `v_2_range` specify the range of the voltages interpolated for v_1 and v_2, respectively. 
    They are each a 2-tuple with the first entry indicating the minimum voltage, in volts, and the
    second entry indicating the maximum voltage, in volts.

    `v_1_n_samples` and `v_2_n_samples` indicate how many points to interpolated. The points are spaced
    linearly using np.linspace. The total number of voltage interpolated will
    be `v_1_n_samples` * `v_2_n_samples`.

    Returns: poles, voltages, distance
    `poles`: 4x3 numpy array where the rows are `p_1`, `p_2`, `p_3`, `p_4`. The poles
    are not guaranteed to be sorted by angle.
    `voltages`: A 4x2 numpy array where the rows, where v[i] indicates the voltages `v_1`, `v_2`
    that must be applied to the crystal to produce the corresponding pole `p_i`.
    `distance`: The sum of the euclidean distance from the near-optimal pole
    p_i to its corresponding optimal pole across the entire set of poles.
    `distance` will be 0 in the case when the near-optimal set of poles matches
    the set of poles.
    """
    def plane_project(lm, points):
        z = np.expand_dims(lm.predict(points[:, :2]), -1)
        p = np.concatenate([points[:, :2], z], axis=-1)
        v1 = p[0]
        v2 = p[np.argmin(np.abs(np.einsum('j,ij->i', p[0], p)))]
        n = np.cross(v1, v2)
        n /= np.linalg.norm(n)
        dist = np.einsum('ij,j->i', points, n)
        projected = points - np.einsum('i,j->ij', dist, n)
        return projected

    voltages = data[:, :2]
    s = data[:, 2:5]
    
    # First step: Find an initial great circle on the data.
    neighbors = NearestNeighbors(n_neighbors=1).fit(s)
    i, j = np.triu_indices(s.shape[0])
    s_p = np.stack([s[i], s[j]], axis=-1)
    u = s_p[:, :, 0]
    v = s_p[:, :, 1]
    t = np.arange(4)*2*np.pi/4
    w = np.cross(np.cross(u, v), u)


    points = np.expand_dims(u, -1)*np.cos(t) + np.expand_dims(w, -1)*np.sin(t)
    points = np.transpose(points, (0, 2, 1))
    t1, t2, t3 = points.shape

    points = points.reshape((t1*t2, t3))
    distances, indices = neighbors.kneighbors(points)
    distances = distances.reshape((t1, t2))
    indices = indices.reshape((t1, t2))
    distances = np.sum(distances, axis=-1)
    min_distance = np.argmin(distances)
    points = points.reshape((t1, t2, t3))

    initial_s = s[indices[min_distance]]
    initial_voltages = voltages[indices[min_distance]]
    initial_distance = distances[min_distance]
    if model is None:
        return initial_s, initial_voltages, initial_distance

    # Second step: Perform additional interpolation
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch must be installed for crystal-fitting.")

    voltages_eval = np.stack(np.meshgrid(
        np.linspace(v_1_range[0], v_1_range[1], v_1_n_samples), 
        np.linspace(v_2_range[0], v_2_range[1], v_2_n_samples)
        ), axis=-1)
    voltages_eval = np.reshape(voltages_eval, (-1, 2))
    voltages_eval = torch.from_numpy(voltages_eval).to(torch.float)
    s_eval = model(voltages_eval).detach().numpy()
    voltages_eval = np.concatenate([voltages, voltages_eval], axis=0)
    s_eval = np.concatenate([s, s_eval], axis=0)


    # We now try to improve the poles using our interpolated points
    # We fit a plane to our initial set of points to find where the candidates
    # for our optimal set of poles would lie. This is just an educated guess and
    # could probably be improved upon in future iterations of this function
    lm = LinearRegression(fit_intercept=False).fit(initial_s[:, 0:2], initial_s[:, 2])
    projected = plane_project(lm, initial_s)
    point_gen_index = np.argmin(np.abs(np.einsum('j,ij->i', projected[0], projected)))
    used_points = [0, point_gen_index]

    r = np.linalg.norm(projected[used_points], axis=-1)
    theta = np.arccos(projected[used_points, 2]/r)
    phi = np.arctan2(projected[used_points, 1], projected[used_points, 0])

    projected = np.stack([
        np.cos(phi)*np.sin(theta),
        np.sin(phi)*np.sin(theta),
        np.cos(theta)
    ], axis=-1)
    u_circ = projected[0]
    v_circ = projected[1]

    # Now that we have determined the plane in which our poles will lie, we find the
    # great circle in this plane. Our optimal set of poles will consist of 4 points from 
    # this great circle
    u_circ = np.expand_dims(u_circ, 0)
    v_circ = np.expand_dims(v_circ, 0)
    t_circ = np.linspace(0, 2*np.pi, int(1e5))
    w_circ = np.cross(np.cross(u_circ, v_circ), u_circ)
    points_circ = np.expand_dims(u_circ, -1)*np.cos(t_circ) + np.expand_dims(w_circ, -1)*np.sin(t_circ)
    points_circ = points_circ[0].T

    interpolated_neighbors = NearestNeighbors(n_neighbors=1).fit(s_eval)
    #point_gen_indices = np.argmin(np.abs(np.einsum('ik,jk->ij', points_circ[i], best_points)), axis=-1)
    u_pole = points_circ
    v_pole = np.expand_dims(u_circ, -1)*np.cos(t_circ + np.pi/2) + np.expand_dims(w_circ, -1)*np.sin(t_circ + np.pi/2)
    v_pole = v_pole[0].T
    t_pole = np.arange(4)*2*np.pi/4
    w_pole = np.cross(np.cross(u_pole, v_pole), u_pole)
    poles = np.expand_dims(u_pole, -1)*np.cos(t_pole) + np.expand_dims(w_pole, -1)*np.sin(t_pole)
    poles = np.transpose(poles, (0, 2, 1))
    t1, t2, t3 = poles.shape

    poles = poles.reshape((t1*t2, t3))
    distances, indices = interpolated_neighbors.kneighbors(poles)
    distances = distances.reshape((t1, t2))
    indices = indices.reshape((t1, t2))
    distances = np.sum(distances, axis=-1)
    min_distance = np.argmin(distances)
    poles = poles.reshape((t1, t2, t3))

    best_s = s_eval[indices[min_distance]]
    best_voltages = voltages_eval[indices[min_distance]]
    best_distance = distances[min_distance]

    if best_distance < initial_distance:
        return best_s, best_voltages, best_distance
    else:
        return initial_s, initial_voltages, initial_distance
