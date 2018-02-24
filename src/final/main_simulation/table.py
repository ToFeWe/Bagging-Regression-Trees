import numpy as np
models = ['friedman','linear']

# Set a table row with placeholders.
table_row_multi = ( r'& {meth} & {mse:.4g} & {var:.4g} & {bias:.4g} &'+
            r' \multirow{{2}}{{*}}{{{rel:.4g}}}\tabularnewline'+
            '\n'
)
table_row_plain = '& {meth} & {mse:.4g} & {var:.4g} & {bias:.4g} &\\tabularnewline\n'
with open('../../bld/out/figures/MainSimulation/'+
            'table_bagging.tex', 'w') as tex_file:
    tex_file.write('\\begin{tabular}{ l l c c c c}\n')
    tex_file.write('\\toprule\n')
    tex_file.write('\\textbf{Model} & \\textbf{Method} & \\textbf{MSE} &'+
                   '\\textbf{Variance} & \\textbf{Bias$^{2}$} & '+
                   '\\textbf{Relative Error}\\tabularnewline\n')
    tex_file.write('\\toprule\n')

    for index, model in enumerate(models):
        if index is not 0:
            tex_file.write('\\midrule\n')
        if model is 'friedman':
            tex_file.write('\\multirow{2}{*}{Friedman \# 1}\n')
        elif model is 'linear':
            tex_file.write('\\multirow{2}{*}{Linear}\n')
        elif model is 'indicator':
            tex_file.write('\\multirow{2}{*}{Indicator}\n')
        
            
        output_bagging = np.load('../../bld/out/analysis/MainSimulation/'+
                                  'simulations_subagging_plot/output_bagging_'+
                                  model+'.npy')
        output_subagging = np.load('../../bld/out/analysis/MainSimulation/'+
                                  'simulations_subagging_plot/output_subagging_'+
                                  model+'.npy')
        tex_file.write(table_row_multi.format(meth='Tree',
                                        mse=output_subagging[-1,0],
                                        var=output_subagging[-1,2],
                                        bias=output_subagging[-1,1],
                                        rel= ((output_subagging[-1,0] -
                                              output_bagging[0]) / output_subagging[-1,0]
                                        )
        ))
        tex_file.write(table_row_plain.format(meth='Bagging',
                                        mse=output_bagging[0],
                                        var=output_bagging[2],
                                        bias=output_bagging[1],
        ))        
    
    tex_file.write('\\bottomrule\n')
    tex_file.write('\\end{tabular}')