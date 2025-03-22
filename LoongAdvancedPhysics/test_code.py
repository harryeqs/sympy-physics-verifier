if __name__ == '__main__':

    namespace = {}
    code = "import sympy as sp\n\n# Define symbols (if needed)\nN = sp.symbols('N\ninteger=True)\n\n# Given constants\n# Neutrino properties\nE_nu = 8e-14  # Joules per neutrino\nflux_density_cm2 = 1e11  # neutrinos per (s cm^2)\n\n# Convert flux density from per cm^2 to per m^2\nflux_density = flux_density_cm2 * 1e4  # neutrinos per (s m^2)\n\n# Inner core properties\nR = 1.2e6  # radius in m (1200 km)\ndensity = 12.8 * 1000  # density in kg/m^3 (12.8 g/cm^3 = 12800 kg/m^3)\nspecific_heat = 0.400 * 1000  # J/(kg K) (0.400 J/gK = 400 J/(kg K))\n\n# Calculate power absorbed by the inner core\n# Neutrino energy flux (J/s m^2)\nenergy_flux = flux_density * E_nu\n\n# Cross-sectional area of the sphere (assuming parallel neutrinos):\narea = sp.pi * R**2\n\n# Total power deposited (in Joules per second)\npower = energy_flux * area\n\n# Calculate the mass of the inner core: volume * density\nvolume = 4/3 * sp.pi * R**3\nmass = density * volume\n\n# Energy required to heat the core by 1 K: mass * specific heat\nenergy_required = mass * specific_heat\n\n# Time required = energy_required / power\ntime_required = energy_required / power\n\n# Express time_required in the form 1eN seconds, solve for N\n# We have time_required = 1eN, so N = log10(time_required)\nN_value = sp.log(time_required, 10)\n\n# For physical insight we evaluate N_value numerically\nN_numeric = sp.N(N_value)\n\n# Round to nearest integer\nN_int = int(round(N_numeric))\n\n# Final result assigned to variable 'result'\nresult = N_int"
    exec(code, namespace)
    result = namespace['result']
    print(result)

    from verifier import PhysicsVerifier
    verifier = PhysicsVerifier()
    result = verifier.execute_code(code)
    print(result)