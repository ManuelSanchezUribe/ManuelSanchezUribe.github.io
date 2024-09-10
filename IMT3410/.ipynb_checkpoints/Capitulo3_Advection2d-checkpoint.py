import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Parameters
N = 100                  # Number of spatial grid points
L = 2 * np.pi            # Domain size
dx = L / N               # Spatial step size
dt = 0.01                # Time step size
T = 2.0                  # Final time
b = np.array([1, 1])     # Advection velocity
n_steps = int(T / dt)    # Number of time steps
snapshot_times = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # Times at which to take snapshots

# Discretized spatial domain
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Initial condition (e.g., a Gaussian)
def initial_condition(x, y):
    return np.exp(-10 * ((x - np.pi)**2 + (y - np.pi)**2))

u = initial_condition(X, Y)

# Periodic boundary helper functions
def periodic_idx(i, N):
    return i % N

# Upwind method for a chunk of the grid
def upwind_step_chunk(u_chunk, b, dx, dt, N, start_idx, end_idx):
    u_new = np.copy(u_chunk)
    
    for i in range(start_idx, end_idx):
        for j in range(N):
            # Indices for upwind scheme
            im = periodic_idx(i - 1, N)
            jm = periodic_idx(j - 1, N)
            
            # Upwind differences
            du_dx = (u_chunk[i, j] - u_chunk[im, j]) / dx if b[0] > 0 else (u_chunk[periodic_idx(i + 1, N), j] - u_chunk[i, j]) / dx
            du_dy = (u_chunk[i, j] - u_chunk[i, jm]) / dx if b[1] > 0 else (u_chunk[i, periodic_idx(j + 1, N)] - u_chunk[i, j]) / dx
            
            # Update u_new
            u_new[i, j] = u_chunk[i, j] - dt * (b[0] * du_dx + b[1] * du_dy)
    
    return u_new

def upwind_step(u, b, dx, dt, N):
    num_chunks = cpu_count()
    chunk_size = (N + num_chunks - 1) // num_chunks  # Ensure all points are covered
    
    chunks = [(u[i*chunk_size:(i+1)*chunk_size, :], b, dx, dt, N, i*chunk_size, min((i + 1) * chunk_size, N)) for i in range(num_chunks)]
    
    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        futures = [executor.submit(upwind_step_chunk, chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6]) for chunk in chunks]
        results = [future.result() for future in as_completed(futures)]
    
    u_new = np.vstack(results)
    return u_new

# Function to perform time-stepping and capture snapshots
def time_stepper(params):
    u, b, dx, dt, N, start_idx, end_idx, snapshot_times, initial_u = params
    current_time = start_idx * dt
    u = np.copy(initial_u)  # Start with the initial condition for each process
    snapshots = []

    for n in range(start_idx, end_idx):
        u = upwind_step(u, b, dx, dt, N)
        current_time += dt
        
        # Capture snapshot if the time is close to one of the snapshot times
        if np.any(np.isclose(current_time, snapshot_times, atol=dt / 2)):
            snapshots.append((current_time, np.copy(u)))

    return snapshots

# Parallel execution of the time-stepping loop
if __name__ == "__main__":
    # Number of available CPU cores
    n_procs = cpu_count()

    # Split the time-stepping process into chunks for parallel processing
    chunk_size = (n_steps + n_procs - 1) // n_procs
    chunks = [(u, b, dx, dt, N, i * chunk_size, min((i + 1) * chunk_size, n_steps), snapshot_times, u) for i in range(n_procs)]

    # Create a pool of worker processes
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        futures = [executor.submit(time_stepper, chunk) for chunk in chunks]
        results = [future.result() for future in as_completed(futures)]

    # Collect and sort snapshots
    snapshots = [snapshot for sublist in results for snapshot in sublist]
    snapshots.sort(key=lambda x: x[0])

    # Plotting the snapshots
    plt.figure(figsize=(15, 10))
    for idx, (time, u_snapshot) in enumerate(snapshots):
        plt.subplot(3, 4, idx + 1)
        plt.contourf(X, Y, u_snapshot, 100, cmap='viridis')
        plt.colorbar()
        plt.title(f'Time = {time:.1f}')

    # Adjust layout and show all plots
    plt.tight_layout()
    plt.show()
