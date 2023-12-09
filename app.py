# Import necessary libraries
from flask import Flask, render_template, request, send_file
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import Descriptors
import io

# Load the SVM model from the file
model_path = 'C:/Users/john.m/Documents/AI project/SVM_model.pkl'  # Full path
with open(model_path, 'rb') as f:
    clf = pickle.load(f)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the input page
@app.route('/input')
def input():
    return render_template('input.html')

# Define the route to handle form submission
@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    # Get the input SMILES from the form
    input_smiles = request.form['input_smiles']

    # Generate and display 2D structure
    img_path = generate_and_display_2d_structure(input_smiles)

    # Calculate solubility
    solubility_result = calculate_solubility(input_smiles)

    # Render the result template with the solubility value and image path
    return render_template('result.html', solubility_result=solubility_result, img_path=img_path)

def generate_and_display_2d_structure(smiles_string):
    molecule = Chem.MolFromSmiles(smiles_string)

    if molecule is not None:
        # Generate 2D structure image
        img = Draw.MolToImage(molecule, size=(300, 300))

        # Save the image to a file (you can adjust the path as needed)
        img_path = 'static/2d_structure.png'
        img.save(img_path)

        return img_path
    else:
        return None

def calculate_solubility(smiles):
    X = calculate_descriptors(smiles)
    molecular_weight = calculate_molecular_weight(smiles)
    X = X.reshape(1, -1)
    predictions = clf.predict(X)
    reversed_value = 10 ** predictions[0]
    gperL = reversed_value * molecular_weight

    return gperL

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES input")

    descriptor_matrix = [
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.RingCount(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.MolMR(mol)
    ]

    molecular_weight = Descriptors.MolWt(mol)

    # Normalize descriptor values by dividing by molecular weight
    descriptor_matrix = np.array(descriptor_matrix) / molecular_weight

    # Append the log of the molecular weight at the end
    descriptor_matrix = np.append(descriptor_matrix, np.log(molecular_weight))

    return descriptor_matrix

def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES input")

    molecular_weight = Descriptors.MolWt(mol)

    return molecular_weight

if __name__ == '__main__':
    app.run(debug=True)