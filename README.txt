============= RENEWABLE ENERGY COMMUNITIES VIABILITY ==============

Filipa Santos (filipamfsantos@ua.pt)
Margarida Santos (mh.santos@ua.pt)
Heitor Leme (heitorcgl@ua.pt)

===================================================================

Work developed throughout the 2024/25 Spring Semester of the Masters in Data Science for Social Sciences @ DCSPT, Universidade de Aveiro, Aveiro, Portugal, as part of the Seminário course.

Our main goal with this work was to consolidate analyses (and above all, a process) through which we could extract, engineer and consolidate public available data on hourly Energy Consumptions throughout Continental Portugal. By crossing this information with hourly data on solar irradiation, we intended to make it possible for anyone to identify whether or not forming a Renewable Solar Energy Community would be viable in their neighbourhoods.

This project is split across a few different folders. In the "data" folder, one may find the files that were either pre-downloaded or pre-processed - mainly due to their sizes and/or API availability. The perc_cp_PTD CSV, which consolidates Postal Code consumptions in relation to the Energy Transformation Posts, was pre-processed on QGIS.

The "notebooks" folder contains two notebooks: one presenting the functions and processes applied in order to consolidate multiplying factors for the whole year, and another with the final code - containing all the necessary functions + an example call case.

The "webapp" folder contains a compartmentalized interactive webapp version of the project, with corresponding Docker, HTML and utility files. The webapp is hosted at https://huggingface.co/spaces/hleme/seminario_app.

The "process_notebooks" contains the intermediate notebooks that were created by us throughout the semester, with some visualization, analysis and (a lot of) uncorrected, unordered code. We don't share this folder with the hopes of its code being easily reproducible - as a matter of fact, it isn't -, but rather in hopes of it possibly being a helpful display of our thought processes.

Finally, this project also contains a PDF file with references and explanation on our process. The file is available in Portuguese.

Feel free to reach out to us in case you have any questions/suggestions/corrections.