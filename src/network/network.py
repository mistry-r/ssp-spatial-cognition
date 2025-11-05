import numpy as np

class Network:
    """
    Container for ensembles and connections.
    
    Manages simulation.
    """
    
    def __init__(self, dt=0.001):
        """
        Initialize network
        """
        self.dt = dt
        self.ensembles = []
        self.connections = []
        self.probes = {}
        self.time = 0.0
        
    def add_ensemble(self, ensemble):
        """
        Add ensemble to network
        """
        self.ensembles.append(ensemble)
        return ensemble
    
    def add_connection(self, connection):
        """
        Add connection to network
        """
        self.connections.append(connection)
        return connection
    
    def probe(self, ensemble, name, what='decoded'):
        """
        Add probe to record data
        """
        self.probes[name] = {
            'ensemble': ensemble,
            'what': what,
            'data': []
        }
    
    def step(self):
        """
        Simulate one timestep.
        """
        # Reset ensemble inputs
        for ens in self.ensembles:
            ens.input_current = np.zeros(ens.n_neurons)
        
        # Update connections
        for conn in self.connections:
            conn.step(self.dt)
        
        # Step ensembles
        for ens in self.ensembles:
            ens.step(self.dt)
        
        # Record probes
        for name, probe in self.probes.items():
            ens = probe['ensemble']
            
            if probe['what'] == 'decoded':
                data = ens.decode()
            elif probe['what'] == 'spikes':
                data = ens.neurons.v  # Membrane potentials (for spike detection)
            elif probe['what'] == 'activities':
                currents = ens.input_current
                data = ens.neurons.rate_approximation(currents)
            
            probe['data'].append(data)
        
        self.time += self.dt
    
    def run(self, duration):
        """
        Run simulation for specified duration
        """
        n_steps = int(duration / self.dt)
        
        for _ in range(n_steps):
            self.step()
    
    def get_probe_data(self, name):
        """
        Get recorded probe data
        """
        return np.array(self.probes[name]['data'])