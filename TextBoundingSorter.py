from dataclasses import dataclass

type BoundingBox = tuple[int,int,int,int]

@dataclass
class TextBoundingSorter:
    '''
    Sorts text bounding boxes from a table to arrange them by columns and rows, to ease OCR to table operations
    - bounding_boxes: list containing all the text bounding boxes correctly extracted from the table image
    - table_columns: dictionary where the keys are the columns of the table ordered from left to right, 
    and the values a list of the text bounding boxes belonging to the column
    '''
    bounding_boxes: list[BoundingBox]
    table_columns: dict[str,list[BoundingBox]]

    def run(self):
        # gets the table with the boxes in the right columns and rows
        ordered_columns = self.order_columns()
        table_rows_per_columns = self.order_rows_within_columns(bounding_boxes=self.bounding_boxes, ordered_columns=ordered_columns)
        return self.get_table_array(rows_per_columns=table_rows_per_columns)

    def order_columns(self) -> dict[str,list[BoundingBox]]:
        # sorts the unordered columns keys in the right order based on x coordinates
        ordered_column_keys = sorted(self.table_columns, key=lambda k: self.table_columns[k][0][0])
        ordered_columns = [self.table_columns[ordered_column_keys[i]] for i in range(len(ordered_column_keys))]
        # sorts the bounding boxes based on their y positions to have them from top to bottom of the table
        return {str(i+1): sorted(ordered_columns[i], key=lambda x: x[1]) for i in range(len(ordered_columns))}
        
    
    def get_mean_box_height(self, bounding_boxes:list[BoundingBox]) -> float:
        bounding_box_heights = [box[3] for box in bounding_boxes]
        return sum(bounding_box_heights) / len(bounding_box_heights)
    
    def order_rows_within_columns(self, bounding_boxes:list[BoundingBox], ordered_columns:dict[str,list[BoundingBox]]) -> dict[str,dict[str,list[BoundingBox]]]:
        '''
        Returns a dictionary which contains, for each column, the bounding boxes corresponding to a sepcific row
        '''
        mean_box_height = self.get_mean_box_height(bounding_boxes=bounding_boxes)
        ordered_rows = {}
        k = 1
        # x and y are the top left coordinates of the box, (x + w), (y + h) are the bottom right ones
        # apply a distance to discriminate if two consecutive boxes in a column are from the same row or not
        for column in ordered_columns.keys():
            column_boxes = ordered_columns[column]
            ordered_rows[column] = {}
            ordered_rows[column][f"{k}"] = [column_boxes[0]]
            if len(column_boxes) == 1: 
                #1 box only, in one row only, attribution by rows is done
                pass
            else:
                for i in range(0,len(column_boxes)-1):
                    x1, y1, w1, h1 = column_boxes[i]
                    bottom_y_1 = y1 + h1
                    x2, y2, w2, h2 = column_boxes[i+1]
                    top_y_2 = y2 
                    #checks whether boxes are consecutive, if so they belong to the same row
                    if abs(top_y_2 - bottom_y_1) < mean_box_height // 2: 
                        ordered_rows[column][f"{k}"].append(column_boxes[i+1])
                    else:
                        k+=1
                        ordered_rows[column][f"{k}"] = [column_boxes[i+1]]
                # sets the row counter back to 1
                k = 1
        return ordered_rows
    
    def get_table_array(self, rows_per_columns:dict[str,dict[str,list[BoundingBox]]]):
        table_array = []
        # all columns should have the same number of rows based on the document structure
        rows = list(rows_per_columns['1'].keys())
        row_numbers = [int(row_nbr) for row_nbr in rows]
        ordered_row_numbers = sorted(row_numbers)
        for i in range(len(ordered_row_numbers)):
            row = []
            key = str(ordered_row_numbers[i])
            row_tupple = [rows_per_columns[column_key][key] for column_key in rows_per_columns.keys()]
            row.append(row_tupple)
            table_array.append(row)
        return table_array



def main():
    bounding_boxes = [(165, 944, 478, 70), (1984, 1699, 723, 75), (170, 1356, 655, 56), (1141, 441, 536, 55), (1981, 169, 455, 73), (164, 2316, 437, 72), (162, 202, 531, 52), (1143, 779, 576, 71), (1151, 1772, 359, 70), (169, 2149, 392, 55), (1148, 2404, 437, 59), (1983, 2167, 811, 79), (1990, 2925, 839, 71), (1139, 191, 578, 54), (164, 2565, 310, 56), (1983, 1780, 504, 74), (1984, 1861, 797, 203), (1983, 1111, 467, 62), (162, 2734, 429, 58), (1981, 1530, 564, 65), (1993, 2765, 190, 57), (1981, 597, 820, 70), (167, 1110, 501, 70), (1148, 2323, 428, 73), (1151, 1692, 434, 56), (1991, 2421, 523, 62), (1983, 1362, 679, 72), (172, 1687, 579, 71), (170, 1522, 522, 70), (1989, 2592, 770, 78), (1142, 610, 510, 55), (1150, 1526, 618, 56), (1147, 1115, 653, 55), (165, 777, 366, 54), (165, 363, 525, 71), (164, 610, 505, 71), (1146, 947, 654, 72), (168, 1190, 287, 54), (1147, 2744, 333, 59), (1140, 360, 312, 54), (1984, 425, 596, 72), (1149, 1361, 548, 55), (1984, 940, 599, 72), (1149, 2154, 488, 59), (1981, 1191, 720, 73), (1994, 2847, 789, 72), (1990, 2339, 638, 71), (1147, 2573, 718, 83), (1984, 767, 759, 75), (1147, 2826, 365, 74), (1984, 342, 404, 71)]
    table_columns = {
        '1': [(162, 202, 531, 52), (162, 2734, 429, 58), (164, 2316, 437, 72), (164, 2565, 310, 56), (164, 610, 505, 71), (165, 944, 478, 70), (165, 777, 366, 54), (165, 363, 525, 71), (167, 1110, 501, 70), (168, 1190, 287, 54), (169, 2149, 392, 55), (170, 1522, 522, 70), (170, 1356, 655, 56), (172, 1687, 579, 71)], 
        '2': [(1139, 191, 578, 54), (1140, 360, 312, 54), (1141, 441, 536, 55), (1142, 610, 510, 55), (1143, 779, 576, 71), (1146, 947, 654, 72), (1147, 1115, 653, 55), (1147, 2744, 333, 59), (1147, 2573, 718, 83), (1147, 2826, 365, 74), (1148, 2323, 428, 73), (1148, 2404, 437, 59), (1149, 1361, 548, 55), (1149, 2154, 488, 59), (1150, 1526, 618, 56), (1151, 1772, 359, 70), (1151, 1692, 434, 56)], 
        '3': [(1981, 169, 455, 73), (1981, 1191, 720, 73), (1981, 1530, 564, 65), (1981, 597, 820, 70), (1983, 2167, 811, 79), (1983, 1780, 504, 74), (1983, 1111, 467, 62), (1983, 1362, 679, 72), (1984, 1861, 797, 203), (1984, 940, 599, 72), (1984, 767, 759, 75), (1984, 342, 404, 71), (1984, 1699, 723, 75), (1984, 425, 596, 72), (1989, 2592, 770, 78), (1990, 2925, 839, 71), (1990, 2339, 638, 71), (1991, 2421, 523, 62), (1993, 2765, 190, 57), (1994, 2847, 789, 72)]
        }
    bounding_box_sorter = TextBoundingSorter(
        bounding_boxes=bounding_boxes,
        table_columns=table_columns
    )
    table_bounding_box_array = bounding_box_sorter.run()
    print(table_bounding_box_array)
  



if __name__ == "__main__":
    main()