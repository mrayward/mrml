from warnings import warn
import re
import os

class Contact:
    def __init__(self, name, email, telephone):
        self.name = name
        self.email = email
        self.telephone = telephone

    def __str__(self):
        val = ''
        val += "name: " + self.name + "\n"

        return val

    def prnt_all(self):
        print('Name: ' + self.name + ' Email: ' + self.email + ' Phone: ' + self.telephone)

    def contact2json(self):
        return '{' + 'name:' + self.name + ',' + 'email:' + self.email + ',' + 'telephone:' + self.telephone + '}'

    def json2contact(self, txt):
        """

        :param txt:
        :return:
        """
        c = txt.strip('{')
        c = c.strip('},\n')
        c = c.strip('}\n')
        properts = c.split(',')
        for prop in properts:
            p = prop.split(':')
            if p[0] == 'name':
                self.name = p[1]
            elif p[0] == 'email':
                self.email = p[1]
            elif p[0] == 'telephone':
                self.telephone = p[1]
            else:
                warn('Unknown property of a contact ', p[0])

    def isin(self, t):

        p = [self.name, self.email, self.telephone]
        tl = t.lower()
        for prop in p:
            if tl in prop.lower():
                return True
        return False


class Agenda:
    def __init__(self):
        self.contact_list = list()
        self.filename = 'agenda.json'

        self.open()

    def open(self):

        if os.path.exists(self.filename):

            txt = open(self.filename, 'r').readlines()

            for txt_chunk in txt[1:-1]:
                c = Contact(None, None, None)
                c.json2contact(txt_chunk)
                self.contact_list.append(c)

    def save(self):
        """

        :return:
        """
        big_js = '{\n'
        for cont in self.contact_list:
            js = cont.contact2json()
            big_js += js + ',\n'
        big_js = big_js[:-1]  # remove the last comma
        big_js += '\n}'

        text_file = open(self.filename, "w")
        text_file.write(big_js)
        text_file.close()

    def add(self, contact: Contact):
        self.contact_list.append(contact)
        self.save()

    def delete_contact(self, contact: Contact):
        self.contact_list.remove(contact)

    def search(self, strng):
        """

        :param strng:
        :return:
        """
        result_list = list()
        for cont in self.contact_list:
            if cont.isin(strng):
                result_list.append(cont)
        return result_list

    def show(self):
        for contact in self.contact_list:
            contact.prnt_all()

    def run(self):
        """

        :return:
        """
        data = ''
        while data != 'exit':

            data = input('How can I help you? ("add", "search", "delete", "show"):')
            print(data)
            if data == 'add':
                nme = input('Please enter the name you would like to add:')
                eml = input('Please enter the email you would like to add:')
                tlf = input('Please enter the telephone number (without spaces or dashes) you would like to add:')
                contact = Contact(nme, eml, tlf)
                self.add(contact)
                print('Your contact has been added!')
            elif data == 'delete':
                inpt = input('Please enter the name of the contact you would like to delete: ')
                lst = self.search(inpt)
                if len(lst)==0:
                    print('No contact was found')
                else:
                    for i, ctact in enumerate(lst):
                        print(i, ctact)
                    num = input('Select the number of the ', inpt, ' to delete:')

                    try:
                        num = int(num)
                        self.delete_contact(lst[num])
                    except:
                        print('Error, try again...')

            elif data == 'search':
                inpt = input('Please enter the name of the contact you would like to search: ')
                lst = self.search(inpt)
                if len(lst) == 0:
                    print('No contact was found')
                else:
                    for i, ctact in enumerate(lst):
                        print(i, ctact)
                    num = input('Select the number of the ' + inpt + ' to display:')
                    try:
                        num = int(num)
                        lst[num].prnt_all()
                    except:
                        print('Error, try again...')

            elif data == 'show':
                self.show()

            else:
                print('Please enter a valid command or "exit", if you wish you quit this application')

if __name__ == "__main__":
    agenda = Agenda()
    agenda.run()