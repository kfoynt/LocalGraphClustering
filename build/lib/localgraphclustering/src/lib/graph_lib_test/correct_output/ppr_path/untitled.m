fptr=fopen('usps_3nn_eps_stats.smat','w');
num_eps=eps_stats.num_eps;
fprintf(fptr,'%d\n',num_eps);
for i=1:num_eps
    fprintf(fptr,'%f %f %f %f %d %d\n',eps_stats.epsilon(i),...
    eps_stats.conds(i),eps_stats.cuts(i),eps_stats.vols(i),...
    eps_stats.setsizes(i),eps_stats.stepnums(i));
end
fclose(fptr);


fptr=fopen('usps_3nn_rank_stats.smat','w');
nrank_changes=rank_stats.nrank_changes;
nrank_inserts=rank_stats.nrank_inserts;
nsteps=rank_stats.nsteps;
size_for_best_cond=rank_stats.size_for_best_cond;
fprintf(fptr,'%d\n%d\n%d\n%d\n',nrank_changes,nrank_inserts,nsteps,size_for_best_cond);
for i=1:nsteps
    fprintf(fptr,'%d %d %d %d %d %d %f %f\n',rank_stats.starts(i),...
    rank_stats.ends(i),rank_stats.nodes(i),rank_stats.deg_of_pushed(i),...
    rank_stats.size_of_solvec(i),rank_stats.size_of_r(i),rank_stats.val_of_push(i),...
    rank_stats.global_bcond(i));
end
fclose(fptr);

fptr=fopen('usps_3nn_bestclus.smat','w');
fprintf(fptr,'%d\n',actual_length);
for i=1:actual_length
    fprintf(fptr,'%d\n',xids(i));
end
fclose(fptr);