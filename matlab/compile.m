clear all;

if isdir('./build'),
    delete('./build/*');
    rmdir('./build');
end

arch=computer;
mac=strcmp(arch,'MACI64') || strcmp(arch,'MACI') || strcmp(arch,'MAC');
windows=strcmp(arch,'PCWIN64') || strcmp(arch,'PCWIN');
linux=strcmp(arch,'GLNXA64') || strcmp(arch,'GLNX86');
sixtyfourbits=strcmp(arch,'MACI64') || strcmp(arch,'GLNXA64') || strcmp(arch,'PCWIN64');

debug=false;
use_multithread=true;
use_64bits_integers=false;

if debug 
   use_multithread=false;
end

out_dir='./build/';
include_dir={'../include','../lib/liblbfgs/include','../lib/matlab/include'};
link_dir={'../lib/liblbfgs/lib/.libs','../build/lib'};
link_lib={'hcrf','lbfgs'};

compile_flags=' -O';
link_flags=' -O ';

compile_opt='../src/matlab/matHCRF.cpp';
for i=1:numel(include_dir),
    compile_opt = sprintf('%s -I%s',compile_opt, include_dir{i});
end

link_opt='';
for i=1:numel(link_dir),
    link_opt = sprintf('%s -L%s', link_opt, link_dir{i});
end
for i=1:numel(link_lib),
    link_opt = sprintf('%s -l%s', link_opt, link_lib{i});
end
link_opt=[link_opt ' LDFLAGS="$LDFLAGS'...
    ' -Wl,-rpath,' pwd '/../lib/liblbfgs/lib/.libs' ...
    ' -Wl,-rpath,' pwd '/../build/lib"'];

if sixtyfourbits
   if debug
      DEFCOMMON='-largeArrayDims -DDEBUG';
   else
      DEFCOMMON='-largeArrayDims -DNDEBUG';
   end
else
   if debug
      DEFCOMMON='-DDEBUG';
   else
      DEFCOMMON='-DNDEBUG';
   end
end


DEFCOMP='';
if use_multithread 
  if (linux || mac)
     compile_flags=[compile_flags ' -fopenmp']; % we assume gcc
     link_opt=[link_opt ' -lgomp'];
  end
end

DEFS=[DEFCOMMON ' ' DEFCOMP];

% WARNING: on Mac OS  mountain lion, you may have to uncomment the line
%add_flag=' -mmacosx-version-min=10.7'
add_flag='';
if mac
   compile_flags=[compile_flags add_flag];
end


fprintf('compilation of: %s\n',compile_opt);
if windows
   compile_opt = [compile_opt ' -outdir ' out_dir, ' ' DEFS ...
       ' ' link_opt ' OPTIMFLAGS="' compile_flags '" ']; 
else
   compile_opt = [compile_opt ' -outdir ' out_dir, ' ' DEFS ...
       ' CXXOPTIMFLAGS="' compile_flags '" LDOPTIMFLAGS="' link_flags '" ' link_opt];
end

args = regexp(compile_opt, '\s+', 'split');
args = args(find(~cellfun(@isempty, args)));
mex(args{:});

copyfile ../src/matlab/hcrfToolbox.m build/
copyfile ../lib/liblbfgs/lib/.libs/* build/
