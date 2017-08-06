
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
        print('Name: '+ self.name + ' Email: ' + self.email + ' Phone: ' + self.telephone)

class Agenda:
    def __init__(self):
        self.contact_list = list()

    def add(self, contact:Contact):
        self.contact_list.append(contact)

    def delete_contact(self, contact:Contact):
        self.contact_list.remove(contact)

    def search(self, strng):
        result_list = list()
        for cont in self.contact_list:
            if strng in cont.name or strng in cont.email or strng in cont.telephone:
                result_list.append(cont)
        return result_list

    def run(self):

        data = ''
        while data != 'exit':

            data = input('How can I help you? ("add", "search", "delete"):')
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

            else:
                print('Please enter a valid command or "exit", if you wish you quit this application')

if __name__ == "__main__":
    agenda = Agenda()
    agenda.run()