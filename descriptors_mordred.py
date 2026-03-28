import pandas as pd
from multiprocessing import freeze_support
from rdkit import Chem

from rdkit.Chem.Crippen import MolMR, MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcLabuteASA, CalcNumHBA, CalcNumHBD

from mordred import Chi, TopologicalCharge, InformationContent, WalkCount, Autocorrelation
from mordred import AcidBase, RingCount, EState, SLogP, MoeType, MolecularDistanceEdge
from mordred import Calculator, descriptors, is_missing


def canonical_smiles(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    return mols, smiles

# rdkit descriptors
# df['mr'] = [MolMR(mol) for mol in df['mol']]
# df['logp'] = [MolLogP(mol) for mol in df['mol']]
# df['mw'] = [MolWt(mol) for mol in df['mol']]
#
# df['balabanJ'] = [BalabanJ(mol) for mol in df['mol']]
# df['bertzCT'] = [BertzCT(mol) for mol in df['mol']]
#
# df['tpsa'] = [CalcTPSA(mol) for mol in df['mol']]
# df['labuteASA'] = [CalcLabuteASA(mol) for mol in df['mol']]
#
# df['numHBA'] = [CalcNumHBA(mol) for mol in df['mol']]
# df['numHBD'] = [CalcNumHBD(mol) for mol in df['mol']]
# df.head()


# mordred descriptors
if __name__ == '__main__':
    # df = pd.read_csv('smiles_energy.csv')
    df = pd.read_csv('generated_mols.csv')
    df.head()

    mol_objs, Canon_SMILES = canonical_smiles(df.SMILES)

    df['SMILES'] = Canon_SMILES
    df['mol'] = mol_objs

    freeze_support()
    calc = Calculator()

    # descriptors registration
    calc.register(Chi.Chi('path', 1, 'dv', False))
    calc.register(Chi.Chi('chain', 7, 'd', False))
    calc.register(TopologicalCharge.TopologicalCharge('raw', 7))
    calc.register(TopologicalCharge.TopologicalCharge('mean', 4))

    calc.register(InformationContent.InformationContent(2))
    calc.register(InformationContent.BondingIC(2))
    calc.register(InformationContent.ComplementaryIC(5))

    calc.register(WalkCount.WalkCount(9, False, True))

    calc.register(Autocorrelation.ATS(1, 'se'))
    calc.register(Autocorrelation.AATS(8, 'd'))
    calc.register(Autocorrelation.AATS(5, 'i'))
    calc.register(Autocorrelation.AATSC(1, 'Z'))
    calc.register(Autocorrelation.GATS(2, 'p'))
    calc.register(Autocorrelation.GATS(5, 'c'))

    calc.register(AcidBase.BasicGroupCount)
    calc.register(RingCount.RingCount(None, False, False, None, True))

    calc.register(EState.AtomTypeEState('count', 'aaCH'))
    calc.register(EState.AtomTypeEState('count', 'aasC'))
    calc.register(EState.AtomTypeEState('count', 'aaN'))
    calc.register(EState.AtomTypeEState('sum', 'ssNH'))
    calc.register(EState.AtomTypeEState('sum', 'sssCH'))

    calc.register(SLogP.SLogP)
    calc.register(MoeType.SlogP_VSA(1))
    calc.register(MoeType.SlogP_VSA(6))
    calc.register(MoeType.SMR_VSA(3))
    calc.register(MolecularDistanceEdge.MolecularDistanceEdge(3, 3, 'C'))

    res_df = calc.pandas(mol_objs)
    result = pd.concat([df, res_df], axis=1, join='outer')
    print(result.head())

    # result.to_csv("data_descriptors_new.csv")
    # result.to_excel('data_descriptors.xlsx', index=False, sheet_name='Descriptors pyrazoles new')

    result.to_csv("test_2_data_descriptors.csv")
    result.to_excel('test_2_data_descriptors.xlsx')

