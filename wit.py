# Upload 177
from collections import OrderedDict
import datetime
from distutils.dir_util import copy_tree
import filecmp
import math
import os
import random
import shutil
import sys

from matplotlib import pyplot as plt
import networkx as nx


class WitRepoNotFoundError(Exception):
    pass


class WitRepoAlreadyExistsError(Exception):
    pass


class WitFilesNotPartOfRepoError(Exception):
    pass


class WitNoFilesToCommitError(Exception):
    pass


class WitUnableToCheckoutError(Exception):
    pass


class WitCommitNotExistError(Exception):
    pass


class WitUnknownCommitError(Exception):
    pass


class WitBranchExistsError(Exception):
    pass


class Wit:
    WIT_DIRS = ("images", "staging_area")
    STAGED_FILES_FILENAME = "added_files"
    COMMIT_ID_LEN = 40
    SHORT_COMMIT_ID_LEN = 6
    COMMIT_MESSAGE_DATE_FORMAT = "%c %z"  # Example: Sat Aug 29 23:04:45 2020 +0300
    REFERENCES_HEAD_LINE = 0
    REFERENCES_MASTER_LINE = 1
    GENESIS_COMMIT_ID = "None"
    NODE_SIZE = 4000
    ACTIVE_BRANCH_FILENAME = "activated.txt"

    def __init__(self, repo_path=""):
        if not repo_path:
            repo_path = os.getcwd()
        self.repo_path = self._get_repo_from_path(repo_path)
        self.wit_path = os.path.join(self.repo_path, ".wit") if self.repo_path else ""
        self.snapshots_path = os.path.join(self.wit_path, "images")
        self.staging_area = os.path.join(self.wit_path, "staging_area")

    def _get_repo_from_path(self, path):
        while path != os.path.dirname(path):
            if os.path.isdir(os.path.join(path, ".wit")):
                return path

            path = os.path.dirname(path)

        return ""

    def set_active_branch(self, branch_name):
        active_branch_path = os.path.join(self.wit_path, self.ACTIVE_BRANCH_FILENAME)
        with open(active_branch_path, "w") as branch_f:
            # no need to check if file exists because we overwrite
            if branch_name:
                branch_f.write(f"{branch_name}\n")

    def get_active_branch(self):
        active_branch_path = os.path.join(self.wit_path, self.ACTIVE_BRANCH_FILENAME)
        with open(active_branch_path) as branch_f:
            return branch_f.read().strip()

    def get_all_branches(self, include_head=False):
        references_path = os.path.join(self.wit_path, "references.txt")
        branches = OrderedDict()
        with open(references_path) as refs_f:
            for line in refs_f:
                branch, commit_id = line.strip().split("=")
                if branch != "HEAD" or (branch == "HEAD" and include_head):
                    branches[branch] = commit_id

        return branches

    def init(self):
        if os.path.exists(self.repo_path):
            raise WitRepoAlreadyExistsError(f"wit repo already exists in {self.repo_path}")

        wit_path = os.path.join(self.repo_path, ".wit")
        os.mkdir(wit_path)
        self.wit_path = wit_path
        self.snapshots_path = os.path.join(self.wit_path, "images")
        self.staging_area = os.path.join(self.wit_path, "staging_area")

        for dirname in self.WIT_DIRS:
            path_to_create = os.path.join(wit_path, dirname)
            os.mkdir(path_to_create)

        self.set_active_branch("master")

    def _find_wit_repo_from_path(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"could not find {path}")

        common_path = os.path.commonpath([self.repo_path, path])
        if len(common_path) < len(self.repo_path):
            raise WitFilesNotPartOfRepoError(f"{path} is not part of {self.repo_path} repo")

        return self.repo_path

    def _add_parent_directories_if_needed(self, files_to_add):

        dir_path = os.path.dirname(files_to_add) if os.path.isfile(files_to_add) else files_to_add

        dirs_to_create = os.path.relpath(dir_path, self.repo_path)
        path_to_create = os.path.join(self.wit_path, "staging_area", dirs_to_create)
        try:
            os.makedirs(path_to_create)
        except FileExistsError:
            pass

        path_to_create = os.path.abspath(path_to_create)
        if os.path.isfile(files_to_add):
            path_to_create = os.path.join(path_to_create, os.path.basename(files_to_add))

        return path_to_create

    def copy_and_overwrite(self, from_path, to_path):
        if os.path.isdir(to_path) and os.path.basename(to_path) != "staging_area":
            shutil.rmtree(to_path)

        if os.path.isdir(from_path):
            shutil.copytree(from_path, to_path)
        else:
            shutil.copy2(from_path, to_path)

    def _is_line_in_file(self, target_line, filepath):
        if not os.path.exists(filepath):
            return False

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if target_line == line:
                    return True

        return False

    def _add_file_to_staged_files(self, filepath):
        staged_files_path = os.path.join(self.wit_path, self.STAGED_FILES_FILENAME)
        if self._is_line_in_file(filepath, staged_files_path):
            return
        with open(staged_files_path, "a") as staged_f:
            staged_f.write(filepath)

    def add(self, files_to_add):
        files_to_add = os.path.abspath(files_to_add)
        repo_path = self._find_wit_repo_from_path(files_to_add)
        if not repo_path:
            raise WitRepoNotFoundError(f"could not find wit repo under {os.path.abspath(files_to_add)}")

        path_to_add_content = self._add_parent_directories_if_needed(files_to_add)
        self.copy_and_overwrite(files_to_add, path_to_add_content)
        self._add_file_to_staged_files(files_to_add)

    def _create_random_commit_id(self):
        allowed_chars = [chr(letter_id) for letter_id in range(ord("a"), ord("f") + 1)] + [str(i) for i in range(10)]
        return "".join(random.choices(allowed_chars, k=self.COMMIT_ID_LEN))

    def _create_commit_snapshot_dir(self):
        commit_id = self._create_random_commit_id()
        commit_path = os.path.join(self.wit_path, "images", commit_id)
        os.mkdir(commit_path)
        return commit_path, commit_id

    def get_head_id(self):
        references_path = os.path.join(self.wit_path, "references.txt")
        if not os.path.exists(references_path):
            return None
        with open(references_path) as f:
            lines = f.readlines()
            head_line = lines[self.REFERENCES_HEAD_LINE].strip()
            return head_line.split("=")[1]

    def get_master_id(self):
        references_path = os.path.join(self.wit_path, "references.txt")
        if not os.path.exists(references_path):
            return None
        with open(references_path) as f:
            lines = f.readlines()
            head_line = lines[self.REFERENCES_MASTER_LINE].strip()
            return head_line.split("=")[1]

    def _create_commit_metadata(self, commit_id, message, merged_parent=None):
        head_id = self.get_head_id()
        if merged_parent:
            parent = ",".join([head_id, merged_parent])
        else:
            parent = head_id
        local_time = datetime.datetime.now().astimezone()
        metadata_path = os.path.join(self.wit_path, f"{commit_id}.txt")
        with open(metadata_path, 'w') as meta_f:
            meta_f.write(f"parent={parent}\n")
            meta_f.write(f"date={local_time.strftime(self.COMMIT_MESSAGE_DATE_FORMAT)}\n")
            meta_f.write(f"message={message}\n")
        return metadata_path

    def _write_to_commit_references(self, commit_id):
        branches = self.get_all_branches()
        active_branch = self.get_active_branch()
        references_path = os.path.join(self.wit_path, "references.txt")
        with open(references_path, 'w') as ref_f:
            ref_f.write(f"HEAD={commit_id}\n")
            for branch_name, branch_commit_id in branches.items():
                if branch_name == active_branch:
                    ref_f.write(f"{branch_name}={commit_id}\n")  # increment active branch head
                else:
                    ref_f.write(f"{branch_name}={branch_commit_id}\n")  # write old branches unchanged

    def _exists_files_to_commit(self):
        staged_files_path = os.path.join(self.wit_path, self.STAGED_FILES_FILENAME)
        return os.path.exists(staged_files_path)

    def _set_no_files_to_commit(self):
        staged_files_path = os.path.join(self.wit_path, self.STAGED_FILES_FILENAME)
        if os.path.exists(staged_files_path):
            os.remove(staged_files_path)

    def commit(self, message, merged_parent=None):
        if not self._exists_files_to_commit():
            raise WitNoFilesToCommitError("there are no files in the staged area, please use wit add, and then wit commit")
        commit_path, commit_id = self._create_commit_snapshot_dir()
        self._create_commit_metadata(commit_id, message, merged_parent)
        staging_path = self.staging_area
        copy_tree(staging_path, commit_path)
        self._write_to_commit_references(commit_id)
        self._set_no_files_to_commit()

    def _get_commit_snapshot_path(self, commit_id):
        images_path = os.path.join(self.wit_path, "images")
        return os.path.join(images_path, commit_id)

    def get_files_to_be_commited(self):
        staged_files_path = os.path.join(self.wit_path, self.STAGED_FILES_FILENAME)
        files_to_be_commited = []
        if not os.path.exists(staged_files_path):
            return files_to_be_commited

        with open(staged_files_path) as staged_f:
            for filepath in staged_f:
                files_to_be_commited.append(filepath.strip())

        return sorted(files_to_be_commited)

    def _are_files_identical(self, filepath1, filepath2):
        if not os.path.exists(filepath1) or not os.path.exists(filepath2):
            return False

        return filecmp.cmp(filepath1, filepath2)

    def get_tracked_files_not_staged_for_commit(self):
        files_not_staged_for_commit = []
        staging_area = self.staging_area

        for root, _, files in os.walk(staging_area):
            for file in files:
                file_abs_path = os.path.join(root, file)
                file_rel_staging_path = os.path.relpath(file_abs_path, staging_area)
                file_repo_path = os.path.join(self.repo_path, file_rel_staging_path)
                if not self._are_files_identical(file_repo_path, file_abs_path):
                    files_not_staged_for_commit.append(file_repo_path)
        return sorted(files_not_staged_for_commit)

    def get_untracked_files(self):
        untracked_files = []
        staging_area = self.staging_area

        for root, dirs, files in os.walk(self.repo_path, topdown=True):
            dirs[:] = [d for d in dirs if d != ".wit"]  # see https://stackoverflow.com/questions/19859840/excluding-directories-in-os-walk
            for file in files:
                file_abs_path = os.path.join(root, file)
                file_rel_repo_path = os.path.relpath(file_abs_path, self.repo_path)
                file_staging_path = os.path.join(staging_area, file_rel_repo_path)
                if not os.path.exists(file_staging_path):
                    untracked_files.append(file_abs_path)
        return sorted(untracked_files)

    def status(self):
        if not self.repo_path:
            raise WitRepoNotFoundError(f"could not find wit repo under {os.path.abspath(os.getcwd())}")

        head_commit_id = self.get_head_id()
        print(f"current HEAD: {head_commit_id} (active branch: {self.get_active_branch()})")

        print("Changes to be committed:")
        files_to_be_commited = self.get_files_to_be_commited()
        for filepath in files_to_be_commited:
            print(os.path.relpath(filepath, os.getcwd()))

        print()

        print("Changes not staged for commit:")
        files_changed_but_not_staged = self.get_tracked_files_not_staged_for_commit()
        for filepath in files_changed_but_not_staged:
            print(os.path.relpath(filepath, os.getcwd()))

        print()

        print("Untracked files:")
        untracked_files = self.get_untracked_files()
        for filepath in untracked_files:
            print(os.path.relpath(filepath, os.getcwd()))

    def commit_exists(self, commit_id):
        supposed_commit_file_path = f"{os.path.join(self.wit_path, commit_id)}.txt"
        return os.path.isfile(supposed_commit_file_path) and len(commit_id) == self.COMMIT_ID_LEN

    def _get_real_commit_id(self, commit_id_or_name):
        branches = self.get_all_branches(include_head=True)
        if commit_id_or_name in branches:
            return branches[commit_id_or_name]
        else:
            if not self.commit_exists(commit_id_or_name):
                raise WitCommitNotExistError(f"commit {commit_id_or_name} does not exist")
            return commit_id_or_name

    def _restore_working_area(self, commit_id):
        commit_image_path = os.path.join(self.snapshots_path, commit_id)

        for root, _, files in os.walk(commit_image_path):
            for file in files:
                file_abs_path = os.path.join(root, file)
                file_rel_image_path = os.path.relpath(file_abs_path, commit_image_path)
                file_repo_path = os.path.join(self.repo_path, file_rel_image_path)
                shutil.copy2(file_abs_path, file_repo_path)

    def _replace_staging_area(self, commit_id):
        commit_image_path = os.path.join(self.snapshots_path, commit_id)
        shutil.rmtree(self.staging_area)  # deletes the directory itself
        os.mkdir(self.staging_area)  # restores empty directory
        copy_tree(commit_image_path, self.staging_area)  # copy content of commit image to staging area

    def _checkout_files(self, commit_id):
        self._restore_working_area(commit_id)
        self._replace_staging_area(commit_id)

    def _update_head(self, commit_id):
        branches = self.get_all_branches()
        references_path = os.path.join(self.wit_path, "references.txt")
        with open(references_path, 'w') as ref_f:
            ref_f.write(f"HEAD={commit_id}\n")
            for branch_name, branch_commit_id in branches.items():
                ref_f.write(f"{branch_name}={branch_commit_id}\n")

    def is_branch(self, branch_name, include_head=False):
        return branch_name in self.get_all_branches()

    def commit_id_or_name_exists(self, commit_id_or_name):
        return self.is_branch(commit_id_or_name, include_head=True) or self.commit_exists(commit_id_or_name)

    def checkout(self, commit_id_or_name):
        if self.get_tracked_files_not_staged_for_commit():
            raise WitUnableToCheckoutError("There are files that contain changes, please commit them first")
        if self.get_files_to_be_commited():
            raise WitUnableToCheckoutError("There are staged files, please commit them first")
        if not self.commit_id_or_name_exists(commit_id_or_name):
            raise WitUnknownCommitError(f"commit {commit_id_or_name} doesn't seem to exist, run wit.py graph --all")

        commit_id = self._get_real_commit_id(commit_id_or_name)
        self._checkout_files(commit_id)
        self._update_head(commit_id)
        if self.is_branch(commit_id_or_name):
            self.set_active_branch(commit_id_or_name)
        else:
            self.set_active_branch(None)

    def _get_all_commits_until_root(self, commit_id):
        current_commit = commit_id
        yield current_commit, "root"
        while current_commit != self.GENESIS_COMMIT_ID:
            commit_metadata_path = os.path.join(self.wit_path, f"{current_commit}.txt")
            with open(commit_metadata_path) as commit_f:
                parents_line = next(commit_f)
                parents_line = parents_line.strip()
                parents_commits = parents_line.split("=")[1]
                for i, parent_commit in enumerate(parents_commits.split(",")):
                    if i == 0:  # main branch parent
                        current_commit = parent_commit
                        yield current_commit, "main_parent"
                    else:
                        yield parent_commit, "merged_parent"

    def _get_short_commit_id(self, commit_id):
        return commit_id[:self.SHORT_COMMIT_ID_LEN]

    def _get_main_parent_commit_id(self, commit_id):
        metadata_path = os.path.join(self.wit_path, f"{commit_id}.txt")
        with open(metadata_path) as meta_f:
            parents_line = next(meta_f).strip()
            main_parent = parents_line.split("=")[1].split(",")[0]
            return main_parent

    def _get_branch_name_in_graph(self, graph, commit_id):
        all_branches = self.get_all_branches(include_head=False)
        commit_ids_to_branch_names = {ci: b_name for b_name, ci in all_branches.items()}
        if commit_id in commit_ids_to_branch_names:
            return commit_ids_to_branch_names[commit_id]

        for node in graph.nodes:
            parent_commit = self._get_main_parent_commit_id(node)
            if parent_commit == commit_id:
                return graph.nodes[node]['branch_name']

    def _build_graph_from_commit(self, commit_id, branch_name, graph=None):
        if not graph:
            commit_graph = nx.DiGraph()
        else:
            commit_graph = graph

        previous_commit_ids = [None, None]
        for current_commit_id, parent_type in self._get_all_commits_until_root(commit_id):
            short_commit = self._get_short_commit_id(current_commit_id)
            if current_commit_id not in commit_graph.nodes:
                actual_branch = branch_name if parent_type == "main_parent" else self._get_branch_name_in_graph(commit_graph, current_commit_id)
                commit_graph.add_node(current_commit_id, visible=False, branch_name=actual_branch, label=short_commit)
            if previous_commit_ids[0]:
                parent_id = previous_commit_ids[0] if parent_type == "main_parent" else previous_commit_ids[1]
                commit_graph.add_edge(parent_id, current_commit_id, visible=False)

            previous_commit_ids[1] = previous_commit_ids[0]
            previous_commit_ids[0] = current_commit_id

        return commit_graph

    def _add_master_positions_to_graph(self, graph):
        start_position = (1, 0)
        end_position = (-1, 0)
        num_nodes = len(graph.nodes)
        delta_x = (start_position[0] - end_position[0]) / num_nodes
        for i, node_label in enumerate(graph.nodes):
            graph.nodes[node_label]['pos'] = (1 - i * delta_x - delta_x / 2, 0)

    def _get_invisible_branch_node_delta(self, in_degree):
        INITIAL_EDGE_ANGLE = math.pi / 4  # 45 degrees
        FINAL_EDGE_ANGLE = math.pi / 2
        INITIAL_EDGE_LENGTH = math.sqrt(0.2 ** 2 + 0.02 ** 2)

        current_edge_angle = FINAL_EDGE_ANGLE - (INITIAL_EDGE_ANGLE * (0.5 ** in_degree))
        current_edge_length = INITIAL_EDGE_LENGTH * (1.5 ** in_degree)
        delta_x = math.cos(current_edge_angle) * current_edge_length
        delta_y = math.sin(current_edge_angle) * current_edge_length
        return delta_x, delta_y

    def _add_branch_name_to_graph(self, branch_name, commit_id, graph):
        delta_x, delta_y = self._get_invisible_branch_node_delta(graph.in_degree(commit_id))

        commit_pos = graph.nodes[commit_id]['pos']
        branch_pos = (commit_pos[0] + delta_x, commit_pos[1] + delta_y)
        edge = (branch_name, commit_id)
        graph.add_edge(*edge, visible=True, label=branch_name)
        graph.nodes[branch_name]['pos'] = branch_pos
        graph.nodes[branch_name]['visible'] = False
        graph.nodes[branch_name]['label'] = branch_name

    def _get_branch_edge_labels_in_graph(self, graph):
        labels = {}
        for node, data in graph.nodes.data():
            if 'is_branch_node' in data:
                commit_node = list(graph[node])[0]
                labels[(node, commit_node)] = node

        return labels

    def _add_all_branches_to_master_graph(self, graph):
        all_branches = self.get_all_branches(include_head=False)
        all_branches.pop("master")  # we already built the master branch
        for branch_name, commit_id in all_branches.items():
            self._build_graph_from_commit(commit_id, branch_name, graph)

    def _add_node_positions_to_graph(self, graph):
        start_position = (1, 0)
        end_position = (-1, 0)
        master_branch_nodes = [node for node, data in graph.nodes.data() if data['branch_name'] == 'master']
        num_master_branch_nodes = len(master_branch_nodes)
        delta_x = (start_position[0] - end_position[0]) / num_master_branch_nodes
        global_delta_y = 0
        for i, node_label in enumerate(master_branch_nodes):
            graph.nodes[node_label]['pos'] = (1 - i * delta_x - delta_x / 2, 0)

        branches = self.get_all_branches()
        branches.pop('master')
        for branch_name, commit_id in branches.items():
            unique_branch_nodes = [node for node, data in graph.nodes.data() if data['branch_name'] == branch_name]
            if len(unique_branch_nodes) > 0:
                global_delta_y -= 0.5
            parent_node = list(graph.out_edges(commit_id))[0][1]
            prev_x, prev_y = graph.nodes[parent_node]['pos']
            for i, node_label in enumerate(unique_branch_nodes):
                graph.nodes[node_label]['pos'] = (prev_x + (i + 1) * delta_x, prev_y + global_delta_y)

    def _add_head_commits_visibilty(self, graph):
        head_commit_id = self.get_head_id()
        current_node = head_commit_id
        while current_node:
            graph.nodes[current_node]["visible"] = True
            neighbors = list(graph.neighbors(current_node))
            current_node = neighbors[0] if neighbors else None

    def _add_node_visibilty_to_graph(self, graph, show_all_branches=False):
        self._add_head_commits_visibilty(graph)

        if show_all_branches:
            for _, node_data in graph.nodes.data():
                node_data['visible'] = True

    def _add_edge_visibility_to_graph(self, graph):
        for u, v in graph.edges:
            if graph.nodes[u]['visible'] and graph.nodes[v]['visible']:
                graph.edges[u, v]['visible'] = True
            else:
                graph.edges[u, v]['visible'] = False

    def _add_branch_nodes_to_graph(self, graph, add_all_branches=False):
        head_id = self.get_head_id()
        all_branches = self.get_all_branches(include_head=True)
        for branch, commit_id in all_branches.items():
            if add_all_branches or commit_id == head_id:
                self._add_branch_name_to_graph(branch, commit_id, graph)

    def graph(self, all_branches=False, plot=True):
        master_graph = self._build_graph_from_commit(self.get_master_id(), "master")
        self._add_all_branches_to_master_graph(master_graph)
        self._add_node_positions_to_graph(master_graph)
        self._add_node_visibilty_to_graph(master_graph, all_branches)
        self._add_edge_visibility_to_graph(master_graph)
        self._add_branch_nodes_to_graph(master_graph, all_branches)

        if plot:
            plt.figure()
            ax = plt.gca()
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            visible_nodes = [node for node, data in master_graph.nodes.data() if data['visible'] is True]
            visible_edges = [(u, v) for u, v, val in master_graph.edges.data('visible') if val is True]

            nx.draw_networkx(master_graph, labels={node: master_graph.nodes[node]['label'] for node in visible_nodes},
                             nodelist=visible_nodes, edgelist=visible_edges,
                             node_size=self.NODE_SIZE, pos=dict(master_graph.nodes.data("pos")))
            nx.draw_networkx_edge_labels(master_graph, pos=dict(master_graph.nodes.data("pos")),
                                         edge_labels={(u, v): label for u, v, label in master_graph.edges.data("label") if label},
                                         verticalalignment='center', label_pos=1)

            plt.show()

        return master_graph

    def branch(self, branch_name):
        if self.is_branch(branch_name):
            raise WitBranchExistsError(f"branch {branch_name} already exists!")

        references_path = os.path.join(self.wit_path, "references.txt")
        with open(references_path, "a") as refs_f:
            refs_f.write(f"{branch_name}={self.get_head_id()}\n")

    def _is_file_exclusively_changed(self, source_path, target_path, lca_path):
        if os.path.exists(source_path) and not os.path.exists(target_path):
            return True

        if self._are_files_identical(target_path, lca_path) and not self._are_files_identical(source_path, target_path):
            return True

        return False

    def _merge_commits(self, source_commit_id, lca_commit):
        staging_area = self.staging_area
        source_snapshot_path = self._get_commit_snapshot_path(source_commit_id)
        lca_snapshot_path = self._get_commit_snapshot_path(lca_commit)

        for root, _, files in os.walk(source_snapshot_path):
            for file in files:
                file_abs_path = os.path.join(root, file)
                file_rel_source_snapshot_path = os.path.relpath(file_abs_path, source_snapshot_path)
                file_lca_snapshot_path = os.path.join(lca_snapshot_path, file_rel_source_snapshot_path)
                file_staging_path = os.path.join(staging_area, file_rel_source_snapshot_path)
                if self._is_file_exclusively_changed(file_abs_path, file_staging_path, file_lca_snapshot_path):
                    self.copy_and_overwrite(file_abs_path, file_staging_path)
                    self._add_file_to_staged_files(file_rel_source_snapshot_path)

    def merge(self, branch_name_or_commit_id):
        if not self.commit_id_or_name_exists(branch_name_or_commit_id):
            raise WitCommitNotExistError(f"commit {branch_name_or_commit_id} does not exist")

        source_commit = self._get_real_commit_id(branch_name_or_commit_id)
        active_branch = self.get_active_branch()
        commits_graph = self.graph(all_branches=True, plot=False)
        lca_commit = nx.algorithms.lowest_common_ancestor(commits_graph.reverse(),
                                                          source_commit,
                                                          self._get_real_commit_id(active_branch))
        self._merge_commits(self._get_real_commit_id(branch_name_or_commit_id), lca_commit)
        self.commit(f"merged {branch_name_or_commit_id} into {active_branch}", merged_parent=source_commit)


def print_usage():
    print("Usage: python wit.py init|add|commit|status|checkout|graph|branch|merge [...]")


def run_args(args):
    if len(args) == 0 or args[0] not in ["init", "add", "commit", "status", "checkout", "graph", "branch", "merge"]:
        print_usage()
        return

    if args[0] == "init":
        wit_repo = Wit()
        try:
            wit_repo.init()
        except WitRepoAlreadyExistsError:
            print(f"wit repo already exists in {wit_repo.repo_path}")
    elif args[0] == "add":
        wit_repo = Wit()
        try:
            wit_repo.add(args[1])
        except WitRepoNotFoundError as e:
            print(e)
    elif args[0] == "commit":
        wit_repo = Wit()
        try:
            wit_repo.commit(args[1])
        except WitNoFilesToCommitError as e:
            print(e)
    elif args[0] == "status":
        wit_repo = Wit()
        try:
            wit_repo.status()
        except WitRepoNotFoundError as e:
            print(e)
    elif args[0] == "checkout":
        wit_repo = Wit()
        try:
            wit_repo.checkout(args[1])
        except (WitRepoNotFoundError, WitCommitNotExistError, WitUnableToCheckoutError, WitUnknownCommitError) as e:
            print(e)
    elif args[0] == "graph":
        wit_repo = Wit()
        try:
            if len(args) == 2 and args[1] == "--all":
                wit_repo.graph(all_branches=True)
            else:
                wit_repo.graph()
        except WitRepoNotFoundError as e:
            print(e)
    elif args[0] == "branch":
        wit_repo = Wit()
        try:
            wit_repo.branch(args[1])
        except (WitRepoNotFoundError, WitBranchExistsError) as e:
            print(e)
    elif args[0] == "merge":
        wit_repo = Wit()
        try:
            wit_repo.merge(args[1])
        except (WitRepoNotFoundError, WitCommitNotExistError) as e:
            print(e)


if __name__ == "__main__":
    run_args(sys.argv[1:])
