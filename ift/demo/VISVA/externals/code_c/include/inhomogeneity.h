void GetFileExtension( char file[], char extension[] );
Scene *ErodeBinScene3( Scene *bin, float radius );
Scene *DilateBinScene3( Scene *bin, float radius );
Scene* InhomogeneityCorrection( Scene *par_in, Scene *par_mask, float par_function, float par_radio, float par_rgrowth, 
			       int par_compression, int par_protocol, int par_threads, int par_verbose );



