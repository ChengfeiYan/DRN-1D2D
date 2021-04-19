const char* docstring=""
"fasta2aln metaclust.fasta metaclust.aln\n"
"    convert FASTA format alignment metaclust.fasta\n"
"    to PSICOV format alignment metaclust.aln\n"
;

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <fstream>

using namespace std;

int fasta2aln(const string infile="-", const string outfile="-")
{
    ifstream fp_in;
    ofstream fp_out;
    if (infile!="-") fp_in.open(infile.c_str(),ios::in);
    if (outfile!="-") fp_out.open(outfile.c_str(),ofstream::out);
    string sequence,line;
    int nseqs=0;
    while ((infile!="-")?fp_in.good():cin.good())
    {
        if (infile!="-") getline(fp_in,line);
        else getline(cin,line);

        if (line.length()==0) continue;
        if (line[0]=='>')
        {
            if (sequence.length()>0)
            {
                if (outfile!="-") fp_out<<sequence<<endl;
                else                cout<<sequence<<endl;
            }
            sequence.clear();
            nseqs++;
        }
        else
            sequence+=line;
    }
    fp_in.close();
    if (outfile!="-") fp_out<<sequence<<endl;
    else                cout<<sequence<<endl;
    fp_out.close();
    sequence.clear();
    return nseqs;
}

int main(int argc, char **argv)
{
    /* parse commad line argument */
    if(argc<2)
    {
        cerr<<docstring;
        return 0;
    }
    string infile=argv[1];
    string outfile=(argc<=2)?"-":argv[2];
    fasta2aln(infile,outfile);
    return 0;
}
