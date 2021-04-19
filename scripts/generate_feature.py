# -*- coding: utf-8 -*-

import os


def A3M_TO_TGT(fasta_file,a3m_file,root_path,tgt_file):
    TGT_path = './bin/TGT_Package'
    os.system('cp %s %s/temp.a3m'%(a3m_file,TGT_path))
    os.system('cp %s %s/temp.fasta'%(fasta_file,TGT_path))
    os.chdir(TGT_path)
    os.system('./A3M_To_TGT -i temp.fasta -I temp.a3m -o temp.tgt')
    os.chdir(root_path)
    Check = os.system ('cp %s/temp.tgt %s'%(TGT_path, tgt_file))
    return Check


def raptor_property(tgt_file, out_path):
    Raptor_file = './bin/Predict_Property/Predict_Property.sh '
    Check = os.system(Raptor_file+'-i %s -o %s'%(tgt_file, out_path))
    return Check


def fasta2aln(fasta_file,aln_file):
    fasta2aln_file = './bin/fasta2aln '

    os.system(fasta2aln_file+fasta_file+' '+aln_file)


def MSA_TO_HHM(msa_file,hhm_file):
    hhmake = './bin/hh-suite/build/bin/hhmake '
    Check = os.system(hhmake+'-i %s -o %s'%(msa_file,hhm_file))
    return Check


def Loadhhm(hhm_file):
    script_file = './bin/LoadHHM.py '
    Check = os.system('python '+script_file+hhm_file)
    return Check


def ccmpred(aln_file,mat_file,ncpu):
    ccmpred_file = './bin/CCMpred/bin/ccmpred'
    params = ' -R -t %s '%(str(ncpu))
    Check = os.system(ccmpred_file+params+aln_file+' '+mat_file)
    return Check


def alnstats(alnfile,alnstats_path,pdb_id):
    sing_outfile = os.path.join(alnstats_path,pdb_id+'.singout')
    pair_outfile = os.path.join(alnstats_path,pdb_id+'.pairout')
    alnstats_file = './bin/alnstats '
    Check = os.system(alnstats_file+alnfile+' '+sing_outfile+' '+pair_outfile)
    return Check



def generate_feature(fasta_file,a3m_file,save_path,root_path,ncpu=1):

    pdb_id = os.path.splitext(os.path.split(a3m_file)[1])[0]
    sub_path = os.path.join(save_path, pdb_id)
    raptor_path = os.path.join(sub_path, 'Raptor')
    alnstats_path = os.path.join(sub_path, 'alnstats')

    tgt_file = os.path.join(sub_path, pdb_id+'.tgt')
    hhm_file = os.path.join(sub_path, pdb_id+'.hhm')
    aln_file = os.path.join(sub_path, pdb_id+'.aln')
    mat_file = os.path.join(sub_path,pdb_id+'.ccmpred')


    for path in [sub_path, alnstats_path]:
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)


    # ACC & SS3 $ SS8
    print('1.1 A2M to TGT')
    Check = A3M_TO_TGT(fasta_file,a3m_file,root_path,tgt_file)
    if Check:
        print('ERROR tgt')
        return 'ERROR'
        quit

    print('1.2 TGT to ACC&SS3')
    Check = raptor_property(tgt_file, raptor_path)
    if Check:
        print('ERROR raptorx')
        return 'ERROR'
        quit
    print('1. ACC&SS3 done')


    print('2.1 reformat MSA')
    # reformat aln a2m a3m
    fasta2aln(a3m_file, aln_file)

    print('2.2 MSA to HHM')
    #MSA to HHM
    Check = MSA_TO_HHM(a3m_file, hhm_file)
    if Check:
        print('ERROR hhmake hhm')
        return 'ERROR'
        quit

    print('2.3 HHM to PSSM')
    Check = Loadhhm(hhm_file)
    if Check:
        print('ERROR load hhm')
        return 'ERROR'
        quit
    print('2. PSSM done')


    print('3 alnstats')
    Check = alnstats(aln_file, alnstats_path, pdb_id)
    if Check:
        print(pdb_id+' ERROR alnstats')
    print('3. alnstats done')


    print('4. ccmpred')
    Check = ccmpred(aln_file, mat_file, ncpu)
    if Check:
        print('ERROR ccmpred')
        return 'ERROR'
        quit
    print('4. CCMpred done')
























